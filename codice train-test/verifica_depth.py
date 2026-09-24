import os
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Carica immagini salvate
rgb_path = os.path.join(PROJECT_ROOT, "dataset_rgbd_maglioncino", "20260109_153911_938455_rgb.png")
depth_path = os.path.join(PROJECT_ROOT, "dataset_rgbd_maglioncino", "20260109_153911_938455_depth.png")

color = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

print("Tipo dati:", depth.dtype)
print("Dimensioni:", depth.shape)
print("Valori min/max:", depth.min(), depth.max())


# RealSense intrinsics (calibrati colore per 640x480)
intr = rs.intrinsics()
intr.width = 640
intr.height = 480
intr.ppx = 325.12787
intr.ppy = 237.36540
intr.fx = 605.93298
intr.fy = 605.52826
intr.model = rs.distortion.none
intr.coeffs = [0,0,0,0,0]

# Converti depth e rgb in point cloud
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(color),
    o3d.geometry.Image(depth),
    depth_scale=1000.0,      # D435i depth in mm → converti a metri
    depth_trunc=5.0,         # max distanza 5 m
    convert_rgb_to_intensity=False
)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy))

o3d.visualization.draw_geometries([pcd])
