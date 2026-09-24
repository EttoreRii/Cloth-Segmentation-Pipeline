import cv2
import numpy as np
import json
from ultralytics import YOLO

# ==========================================
# 1. COORDINATE EXTRACTION LOGIC
# ==========================================
def get_avg_y_per_x(mask, depth_img=None):
    """
    Computes the average Y-coordinate for each unique X-coordinate in the mask.
    If depth_img is provided, extracts the Z-coordinate as the median of all
    valid depth samples in that column of the mask (robust to pixel-level noise).

    Returns:
        tuple: ({x: avg_y, ...}, {x: z, ...} or None)
    """
    points = cv2.findNonZero(mask)
    if points is None or len(points) == 0:
        return {}, None

    points = points.squeeze()
    if points.ndim == 1:
        x, y = int(points[0]), int(points[1])
        z = float(depth_img[y, x]) if depth_img is not None else None
        return {x: float(y)}, ({x: z} if z is not None else None)

    y_sums = {}
    counts = {}
    z_values = {}

    for x, y in points:
        x_val = int(x)
        y_val = float(y)
        if x_val not in counts:
            y_sums[x_val] = y_val
            counts[x_val] = 1
        else:
            y_sums[x_val] += y_val
            counts[x_val] += 1

        if depth_img is not None:
            z_raw = float(depth_img[int(y), x_val])
            if z_raw > 0:
                z_values.setdefault(x_val, []).append(z_raw)

    avg_y_per_x = {x: y_sums[x] / counts[x] for x in y_sums}

    z_per_x = None
    if depth_img is not None:
        z_per_x = {}
        for x in avg_y_per_x:
            vals = z_values.get(x, [])
            z_per_x[x] = float(np.median(vals)) if vals else 0.0

    return avg_y_per_x, z_per_x

def extract_start_end_points(mask, depth_img=None, num_points=10):
    """
    Extracts equidistant points along the X-axis from a binary mask.
    Uses Y-averaging (and Z-averaging if depth_img provided) for each X.
    Used for Hem.
    
    Returns:
        dict: {"points": [[x, y, z], ...]} or None if no points found
    """
    avg_y_dict, avg_z_dict = get_avg_y_per_x(mask, depth_img)
    if not avg_y_dict:
        return None
    
    sorted_x = sorted(avg_y_dict.keys())
    min_x, max_x = sorted_x[0], sorted_x[-1]
    
    # If we have very few unique X, just return what we have
    if len(sorted_x) < num_points:
        points = []
        for x in sorted_x:
            p = [int(x), int(round(avg_y_dict[x]))]
            if avg_z_dict: p.append(float(avg_z_dict[x]))
            points.append(p)
        return {"points": points}

    # Sample num_points equidistant points between min_x and max_x
    sampled_points = []
    for i in range(num_points):
        target_x = min_x + (max_x - min_x) * i / (num_points - 1)
        target_x = int(round(target_x))
        actual_x = min(sorted_x, key=lambda x: abs(x - target_x))
        
        p = [int(actual_x), int(round(avg_y_dict[actual_x]))]
        if avg_z_dict: p.append(float(avg_z_dict[actual_x]))
        sampled_points.append(p)
        
    return {"points": sampled_points}

def extract_cuff_points(mask, depth_img=None):
    """
    Extracts start (leftmost) and end (rightmost) points from a binary mask.
    Uses Y-averaging for each X.
    
    Returns:
        dict: {"start": [x, y, z], "end": [x, y, z]} or None if no points found
    """
    avg_y_dict, avg_z_dict = get_avg_y_per_x(mask, depth_img)
    if not avg_y_dict:
        return None
    
    sorted_x = sorted(avg_y_dict.keys())
    start_x = sorted_x[0]
    end_x = sorted_x[-1]
    
    start_point = [int(start_x), int(round(avg_y_dict[start_x]))]
    if avg_z_dict: start_point.append(float(avg_z_dict[start_x]))
    
    end_point = [int(end_x), int(round(avg_y_dict[end_x]))]
    if avg_z_dict: end_point.append(float(avg_z_dict[end_x]))
    
    return {"start": start_point, "end": end_point}

def extract_collar_curve(mask, depth_img=None, num_points=10):
    """
    Extracts points along the centerline curve of the collar mask.
    
    Returns:
        dict: {
            "left": [x, y, z],
            "right": [x, y, z],
            "curve_points": [[x, y, z], ...]
        }
    """
    avg_y_dict, avg_z_dict = get_avg_y_per_x(mask, depth_img)
    if not avg_y_dict:
        return None
    
    sorted_x = sorted(avg_y_dict.keys())
    left_x = sorted_x[0]
    right_x = sorted_x[-1]
    
    left_point = [int(left_x), int(round(avg_y_dict[left_x]))]
    if avg_z_dict: left_point.append(float(avg_z_dict[left_x]))
    
    right_point = [int(right_x), int(round(avg_y_dict[right_x]))]
    if avg_z_dict: right_point.append(float(avg_z_dict[right_x]))
    
    # Sample num_points equidistant points
    curve_points = []
    if len(sorted_x) < num_points:
        xs = sorted_x
    else:
        xs = []
        for i in range(num_points):
            target_x = left_x + (right_x - left_x) * i / (num_points - 1)
            target_x = int(round(target_x))
            xs.append(min(sorted_x, key=lambda x: abs(x - target_x)))
            
    for x in xs:
        p = [int(x), int(round(avg_y_dict[x]))]
        if avg_z_dict: p.append(float(avg_z_dict[x]))
        curve_points.append(p)
            
    return {
        "left": left_point,
        "right": right_point,
        "curve_points": curve_points
    }

# ==========================================
# 2. MASK FILTERING LOGIC
# ==========================================
def calculate_iou(mask1, mask2):
    """
    Calculates Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def filter_overlapping_masks(masks, classes, scores, iou_threshold=0.5):
    """
    Filters overlapping masks of the same class, keeping the one with highest score.
    
    Args:
        masks: List of binary masks (numpy arrays)
        classes: List of class IDs
        scores: List of confidence scores
        iou_threshold: Threshold for IoU to consider masks overlapping
        
    Returns:
        indices: List of indices to keep
    """
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while indices:
        current_idx = indices.pop(0)
        keep.append(current_idx)
        
        remove_indices = []
        for i, idx in enumerate(indices):
            # Check if same class
            if classes[current_idx] != classes[idx]:
                continue
                
            # Check IoU
            iou = calculate_iou(masks[current_idx], masks[idx])
            if iou > iou_threshold:
                remove_indices.append(i)
                
        # Remove overlapping masks (since we keeping the highest score one)
        for i in sorted(remove_indices, reverse=True):
            indices.pop(i)
            
    return sorted(keep)

# ==========================================
# 3. INFERENCE CLASS
# ==========================================
class SweaterDetector:
    def __init__(self, model_path="sweater_segmentation\\yolo_run\\weights\\best.pt", 
                 fx=605.93298, fy=605.52826, ppx=325.12787, ppy=237.36540,
                 near=0.3, far=1000.0):
        # Default intrinsics for Intel RealSense D435i Color Stream at 640x480 (calibrated):
        #   fx = 605.93298
        #   fy = 605.52826
        # Load the YOLO model
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        # Camera Intrinsics
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        # Camera Depth Clipping Planes
        self.near = near
        self.far = far
        
    def process_image(self, image_path, depth_path=None, conf_threshold=0.25, override_z=None):
        """
        Runs inference and post-processing.
        Args:
            image_path: Path to RGB image.
            depth_path: Path to depth image/bin.
            conf_threshold: YOLO confidence threshold.
            override_z: If provided, uses this fixed Z value (in mm) for all points.
        """
        # Run inference
        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        
        # Prepare output structure
        final_data = {}
        
        # Load images
        orig_img = cv2.imread(image_path)
        h, w = orig_img.shape[:2]
        
        depth_img = None
        if depth_path:
            if depth_path.lower().endswith('.bin'):
                # Load Unity binary float32 depth
                depth_data = np.fromfile(depth_path, dtype=np.float32)
                try:
                    # Try to reshape to image dimensions
                    depth_img = depth_data.reshape((h, w))
                except ValueError:
                    # Fallback or error
                    print(f"Warning: Binary depth file size {len(depth_data)} does not match image dimensions {h}x{w}.")
                    return final_data, orig_img
                
                # Unity GetPixels() is bottom-up, OpenCV is top-down
                depth_img = np.flipud(depth_img)
               
                
                # The fixed Unity C# script now exports depth directly in linear meters
                depth_img = depth_img * 1000.0  # Convert meters to mm
            else:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            # Ensure depth is 2D (extract first channel if 3D)
            if depth_img is not None and len(depth_img.shape) == 3:
                depth_img = depth_img[:, :, 0]

            # Ensure depth is same size as color
            if depth_img is not None and depth_img.shape[:2] != (h, w):
                depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)

        # Check if masks detected
        if results.masks is None:
            print("No masks detected.")
            return final_data, orig_img
            
        # Get masks and class IDs
        masks_raw = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        scores = results.boxes.conf.cpu().numpy()
        
        # Filter overlapping masks
        keep_indices = filter_overlapping_masks(masks_raw, classes, scores, iou_threshold=0.5)
        print(f"Detected {len(masks_raw)} masks, keeping {len(keep_indices)} after filtering.")
        
        for i in keep_indices:
            class_id = int(classes[i])
            class_name = results.names[class_id]
            
            # Resize mask to original image size
            mask_cpu = masks_raw[i]
            mask_resized = cv2.resize(mask_cpu, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized * 255).astype(np.uint8)
            
            # Extract coordinates based on Class
            raw_coords = None
            if class_name == "Hem":
                raw_coords = extract_start_end_points(mask_binary, depth_img)
            elif class_name == "Cuff":
                raw_coords = extract_cuff_points(mask_binary, depth_img)
            elif class_name == "Collar":
                raw_coords = extract_collar_curve(mask_binary, depth_img)
            
            if raw_coords:
                # Transform to spatial using the mask for robustness, or use override_z
                spatial_coords = self.transform_instance_to_spatial(class_name, raw_coords, depth_img, mask_binary, override_z=override_z)
                if spatial_coords:
                    if class_name not in final_data:
                        final_data[class_name] = []
                    final_data[class_name].append(spatial_coords)
            
            # VISUALIZATION: Draw extracted points based on raw_coords
            if raw_coords:
                if class_name == "Hem":
                    # Draw sampled points
                    pts = raw_coords["points"]
                    for pt in pts:
                        # Draw only (x, y)
                        cv2.circle(orig_img, (int(pt[0]), int(pt[1])), 5, (255, 0, 255), -1)  # Magenta for hem points
                    
                    # Draw polyline
                    if len(pts) > 1:
                        # Use only (x, y) for polyline
                        pts_array = np.array([pt[:2] for pt in pts], np.int32)
                        cv2.polylines(orig_img, [pts_array], False, (255, 255, 0), 2)

                elif class_name == "Cuff":
                    # Draw start and end points (2 points)
                    start_pt = (int(raw_coords["start"][0]), int(raw_coords["start"][1]))
                    end_pt = (int(raw_coords["end"][0]), int(raw_coords["end"][1]))
                    
                    cv2.circle(orig_img, start_pt, 5, (0, 255, 0), -1)  # Green for start
                    cv2.circle(orig_img, end_pt, 5, (0, 0, 255), -1)    # Red for end
                    
                    # Draw line connecting start -> end
                    cv2.line(orig_img, start_pt, end_pt, (255, 255, 0), 2)  # Cyan line
                    
                elif class_name == "Collar":
                    # Draw the curve points
                    curve_pts = raw_coords["curve_points"]
                    
                    # Draw the curve as a polyline
                    if len(curve_pts) > 1:
                        # Use only (x, y) for polyline
                        curve_array = np.array([pt[:2] for pt in curve_pts], np.int32)
                        cv2.polylines(orig_img, [curve_array], False, (0, 255, 255), 3)  # Yellow thick line
                        
                        # Draw the sampled points
                        for pt in curve_pts:
                            cv2.circle(orig_img, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)
                
                # Draw Label
                # Extract (x, y) from the first point available
                if "start" in raw_coords:
                    label_pos = raw_coords["start"]
                elif "left" in raw_coords:
                    label_pos = raw_coords["left"]
                elif "points" in raw_coords and len(raw_coords["points"]) > 0:
                    label_pos = raw_coords["points"][0]
                else:
                    label_pos = [10, 10]
                
                cv2.putText(orig_img, class_name, (int(label_pos[0]), int(label_pos[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
        return final_data, orig_img

    def transform_instance_to_spatial(self, class_name, raw_coords, depth_img, mask_binary, override_z=None):
        """
        Transforms coordinates for a single instance using mask-aware depth sampling.
        """
        h, w = depth_img.shape[:2]
        ppx, ppy = self.ppx, self.ppy
        fx, fy = self.fx, self.fy
        
        # Stability anchor: calculate median depth of the entire mask, 
        # but EXCLUDE pixels that are clearly floor (e.g. Z > 1200mm)
        mask_pixels = depth_img[mask_binary == 255]
        valid_mask_pixels = mask_pixels[(mask_pixels > 100) & (mask_pixels < 1200)]
        
        if override_z is not None:
            mask_median_z = float(override_z)
        elif valid_mask_pixels.size == 0:
            # If no pixel is < 1200, maybe the object is actually further? 
            # Fallback to absolute median, but this shouldn't happen for the sweater
            mask_median_z = np.median(mask_pixels)
        else:
            mask_median_z = np.median(valid_mask_pixels)

        def convert_point(pt):
            if pt is None: return None
            x_pix, y_pix = pt[0], pt[1]

            # Use per-point Z sampled from depth_img (pt[2], already in mm).
            # Fall back to mask median if the sample is missing or an outlier
            # (e.g. a wrinkle edge pixel, a small hole, or a clipped value).
            z_fixed = mask_median_z  # default fallback
            if override_z is None and len(pt) >= 3 and pt[2] is not None:
                z_sample = float(pt[2])
                # Accept the per-point value only if it is physically plausible
                # (> 50 mm) and close enough to the mask median (±200 mm).
                if z_sample > 50.0 and abs(z_sample - mask_median_z) < 200.0:
                    z_fixed = z_sample

            X = (x_pix - ppx) * z_fixed / fx
            Y = (y_pix - ppy) * z_fixed / fy

            # X = ((x_pix - ppx) * z_fixed / fx) * 0.08
            # Y = ((y_pix - ppy) * z_fixed / fy) * 0.08
            return [round(float(X), 2), round(float(Y), 2), round(float(z_fixed), 2)]

        spatial_inst = {}
        for key, val in raw_coords.items():
            if isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], (list, tuple)): # List of points 
                    spatial_inst[key] = [convert_point(p) for p in val]
                else: # Single point
                    spatial_inst[key] = convert_point(val)
            
        return spatial_inst
    
    def save_coordinates_json(self, coordinates, output_path="robot_coordinates.json"):
        """
        Save extracted coordinates to JSON file for robot control.
        
        Args:
            coordinates: Dictionary of coordinates returned by process_image
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(coordinates, f, indent=4)
        print(f"Coordinates saved to {output_path}")


if __name__ == "__main__":
    # Example Usage
    detector = SweaterDetector("sweater_segmentation/yolo_run/weights/best.pt")
    
    # Test on an image
    #IMAGE_PATH = "dataset_rgbd_maglioncino\\20260109_153431_498491_rgb.png"
    IMAGE_PATH = "dataset_rgbd_maglioncino/20260806_130310_267623_rgb.png"

    print(f"Processing image: {IMAGE_PATH}")
    coordinates, vis_img = detector.process_image(IMAGE_PATH, conf_threshold=0.25)
    
    # Save visualization
    cv2.imwrite("yolo_inference_result.jpg", vis_img)
    print("Visualization saved to yolo_inference_result.jpg")
    
    # Save coordinates to JSON for robot control
    detector.save_coordinates_json(coordinates, "robot_coordinates.json")
    
    # Print extracted coordinates
    print("\nExtracted Coordinates:")
    for class_name, coords_list in coordinates.items():
        print(f"\n{class_name}:")
        for i, coords in enumerate(coords_list):
            print(f"  Instance {i+1}:")
            if coords:
                for key, value in coords.items():
                    print(f"    {key}: {value}")
            else:
                print("    No coordinates extracted")
