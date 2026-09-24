import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# 📁 CARTELLA DI SALVATAGGIO (modifica qui)
save_dir = r"dataset_rgbd_maglioncino"
os.makedirs(save_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()


#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Filtri di post-processing
spatial   = rs.spatial_filter()
temporal  = rs.temporal_filter()
hole_fill = rs.hole_filling_filter()
align     = rs.align(rs.stream.color)

# Warm-up (frame scartati, servono a stabilizzare auto-exposure/depth)
for _ in range(30):
    pipeline.wait_for_frames()

# Cattura N frame filtrati e fai la mediana pixel-per-pixel della depth
N = 15
depth_frames_accum = []
color_frame = None

for _ in range(N):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()  # tieni l'ultimo per l'RGB

    if not depth_frame or not color_frame:
        continue

    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_fill.process(depth_frame)

    depth_frames_accum.append(np.asanyarray(depth_frame.get_data()))

if not depth_frames_accum or color_frame is None:
    pipeline.stop()
    raise RuntimeError("Frame non validi durante l'acquisizione")

depth = np.median(depth_frames_accum, axis=0).astype(np.uint16)
color = np.asanyarray(color_frame.get_data())

# Stampa intrinseci colore attivi prima di fermare la pipeline
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()
print("--- INTRINSECI COLORE REALSENSE (640x480) ---")
print(f"fx: {color_intrinsics.fx}")
print(f"fy: {color_intrinsics.fy}")
print(f"ppx: {color_intrinsics.ppx}")
print(f"ppy: {color_intrinsics.ppy}")
print("--------------------------------------------")

# 💾 Salvataggio con path
rgb_path   = os.path.join(save_dir, f"{ts}_rgb.png")
depth_path = os.path.join(save_dir, f"{ts}_depth.png")

cv2.imwrite(rgb_path, color)
cv2.imwrite(depth_path, depth)  # 16-bit depth

pipeline.stop()

print(f"📸 RGB salvata in: {rgb_path}")
print(f"📏 Depth salvata in: {depth_path}")
print(depth.min(), depth.max())