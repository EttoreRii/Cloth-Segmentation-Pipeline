import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime



ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# 📁 CARTELLA DI SALVATAGGIO (modifica qui)
save_dir = r"dataset_rgbd_maglioncino"

# Crea la cartella se non esiste
os.makedirs(save_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Warm-up
for _ in range(30):
    pipeline.wait_for_frames()

# Scatto
frames = pipeline.wait_for_frames()

# Stampa intrinseci colore attivi prima di fermare la pipeline
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()
print("--- INTRINSECI COLORE REALSENSE (640x480) ---")
print(f"fx: {color_intrinsics.fx}")
print(f"fy: {color_intrinsics.fy}")
print(f"ppx: {color_intrinsics.ppx}")
print(f"ppy: {color_intrinsics.ppy}")
print("--------------------------------------------")

align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)

color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

if not color_frame or not depth_frame:
    pipeline.stop()
    raise RuntimeError("Frame non valido")

color = np.asanyarray(color_frame.get_data())
depth = np.asanyarray(depth_frame.get_data())

# 💾 Salvataggio con path
# rgb_path   = os.path.join(save_dir, "rgb.png")
# depth_path = os.path.join(save_dir, "depth.png")
rgb_path   = os.path.join(save_dir, f"{ts}_rgb.png")
depth_path = os.path.join(save_dir, f"{ts}_depth.png")


cv2.imwrite(rgb_path, color)
cv2.imwrite(depth_path, depth)  # 16-bit depth

pipeline.stop()

print(f"📸 RGB salvata in: {rgb_path}")
print(f"📏 Depth salvata in: {depth_path}")
print(depth.min(), depth.max())
