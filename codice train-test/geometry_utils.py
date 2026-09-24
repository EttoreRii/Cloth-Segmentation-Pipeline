import os
import cv2
import numpy as np

def extract_top_boundary(mask):
    """
    Extracts the 'top' boundary of a binary mask.
    For a sweater laid out, this usually corresponds to the lowest y-coordinate 
    for each x-coordinate present in the mask.
    
    Args:
        mask (np.ndarray): Binary mask (0 and 1/255).
        
    Returns:
        list: List of (x, y) coordinates representing the top boundary.
    """
    # Ensure mask is binary
    points = cv2.findNonZero(mask)
    if points is None:
        return []
    
    points = points.squeeze() # (N, 2)
    
    # Sort by x then y
    # For top boundary: for each unique X, we want the minimum Y.
    
    # Using a dictionary to store min Y for each X
    min_y_per_x = {}
    
    for x, y in points:
        if x not in min_y_per_x:
            min_y_per_x[x] = y
        else:
            if y < min_y_per_x[x]:
                min_y_per_x[x] = y
                
    # Convert back to sorted list of points
    sorted_x = sorted(min_y_per_x.keys())
    top_boundary = [(x, min_y_per_x[x]) for x in sorted_x]
    
    return top_boundary

def extract_bottom_boundary(mask):
    """
    Extracts the 'bottom' boundary of a binary mask.
    For the crew neck, if we want the 'contour', we might need the whole contour,
    but user asked for 'lower border or contour'. 
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        list: List of (x, y) coordinates.
    """
    # If the user wants the full contour for the neck:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    # Assume the largest contour is the object
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Return as list of (x, y)
    return [tuple(p[0]) for p in largest_contour]

def associate_depth(coordinates, depth_map):
    """
    Associates depth values with 2D coordinates.
    
    Args:
        coordinates (list): List of (x, y) tuples.
        depth_map (np.ndarray): Depth image (same resolution as RGB).
        
    Returns:
        list: List of (x, y, z) tuples.
    """
    coordinates_3d = []
    h, w = depth_map.shape[:2]
    
    for x, y in coordinates:
        # Check bounds
        if 0 <= y < h and 0 <= x < w:
            z = depth_map[y, x]
            coordinates_3d.append((x, y, z))
        else:
            coordinates_3d.append((x, y, 0)) # Default or skip
            
    return coordinates_3d

def visualize_coordinates_3d(rgb_path, depth_path, coordinates_json_path):
    """
    Visualizes the point cloud and the extracted coordinates in 3D.
    """
    import open3d as o3d
    import json
    import pyrealsense2 as rs

    # 1. Load images
    color = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if color is None or depth is None:
        print(f"Error loading images: color={rgb_path}, depth={depth_path}")
        return

    # 2. Configura intrinseci (calibrati colore)
    h, w = depth.shape[:2]
    if w == 1920 and h == 1080:
        fx, fy, ppx, ppy = 980.34, 980.34, 968.329, 524.537
    elif w == 1280 and h == 720:
        fx, fy, ppx, ppy = 908.89948, 908.29236, 647.69177, 356.04810
    else: # Default 640x480
        fx, fy, ppx, ppy = 605.93298, 605.52826, 325.12787, 237.36540
        
    intr = o3d.camera.PinholeCameraIntrinsic(
        w, h, fx, fy, ppx, ppy
    )

    # 3. Create Point Cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth),
        depth_scale=1000.0,
        depth_trunc=5.0,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # 4. Load coordinates
    if not os.path.exists(coordinates_json_path):
        print(f"Error: Coordinates file not found at {coordinates_json_path}")
        return

    with open(coordinates_json_path, 'r') as f:
        coords_data = json.load(f)

    # 5. Create 3D markers for points
    geometries = [pcd]
    
    # Colori per i diversi componenti
    colors = {
        "Collar": [1, 1, 0], # Giallo
        "Cuff": [0, 1, 0],   # Verde
        "Hem": [1, 0, 1]    # Magenta
    }

    def add_sphere(pos, color):
        # pos = [x, y, z] in pixel and mm
        # Dobbiamo convertire x,y pixel in coordinate 3D usando gli intrinseci
        # z_m = z_mm / 1000.0
        z_m = pos[2] / 1000.0
        if z_m <= 0: return # Skip invalid depth
        
        x_m = pos[0] / 1000.0
        y_m = pos[1] / 1000.0
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005) # 5mm sphere
        sphere.paint_uniform_color(color)
        sphere.translate([x_m, y_m, z_m])
        geometries.append(sphere)

    for class_name, instances in coords_data.items():
        color_marker = colors.get(class_name, [0, 0, 1]) # Default blue
        for inst in instances:
            if not inst: continue
            
            # Extract points based on structure
            if "points" in inst: # Hem
                for p in inst["points"]:
                    if len(p) == 3: add_sphere(p, color_marker)
            elif "start" in inst: # Cuff
                if len(inst["start"]) == 3: add_sphere(inst["start"], color_marker)
                if len(inst["end"]) == 3: add_sphere(inst["end"], color_marker)
            elif "curve_points" in inst: # Collar
                for p in inst["curve_points"]:
                    if len(p) == 3: add_sphere(p, color_marker)

    # 6. Draw
    print("Visualizing 3D. Markers color: Yellow=Collar, Green=Cuff, Magenta=Hem")
    o3d.visualization.draw_geometries(geometries)
