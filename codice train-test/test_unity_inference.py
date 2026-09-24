import cv2
import os
from yolo_inference import SweaterDetector, resolve_model_path, PROJECT_ROOT

def test_unity():
    # Paths
    MODEL_PATH = resolve_model_path()
    RGB_PATH = os.path.join(PROJECT_ROOT, "test_unity", "maglioncino_bianco.jpeg")
    DEPTH_BIN_PATH = os.path.join(PROJECT_ROOT, "test_unity", "maglioncino_camera_depth6.bin")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "risultati rete")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    OUTPUT_JSON = os.path.join(RESULTS_DIR, "robot_coordinates_unity.json")
    OUTPUT_IMG = os.path.join(RESULTS_DIR, "yolo_inference_unity_result.jpg")
    
    print("--- Unity Inference Test ---")
    print(f"RGB: {RGB_PATH}")
    print(f"Depth Bin: {DEPTH_BIN_PATH}")
    
    # 1. Initialize detector with Unity-specific intrinsics
    # Unity Camera: FOV=60 VERTICAL, resolution 640x480
    # For a square-pixel camera, fx = fy = (Height/2) / tan(vFOV/2)
    # fx = fy = 240 / tan(30°) = 415.69
    detector = SweaterDetector(MODEL_PATH, fx=415.69, fy=415.69, ppx=320.0, ppy=240.0)
    
    # 2. Run inference
    print("\nProcessing Unity data with Fixed Z = 530mm...")
    coordinates, vis_img = detector.process_image(RGB_PATH, depth_path=DEPTH_BIN_PATH, conf_threshold=0.25, override_z=515.0)
    
    # 3. Save results
    if coordinates:
        detector.save_coordinates_json(coordinates, OUTPUT_JSON)
        cv2.imwrite(OUTPUT_IMG, vis_img)
        print("\nSuccess!")
        print(f"Visualization saved to {OUTPUT_IMG}")
        print(f"Coordinates saved to {OUTPUT_JSON}")
        
        # Print NumPy array format for copy-pasting
        print("\n--- Processed Coordinates (NumPy Format) ---\n")
        
        # 1. Cuffs (Polsini)
        cuffs = coordinates.get("Cuff", [])
        if cuffs:
            # Sort cuffs by X coordinate to distinguish left (sx) and right (dx)
            cuffs.sort(key=lambda c: c["start"][0])
            for i, prefix in enumerate(["sx", "dx"]):
                if i < len(cuffs):
                    c = cuffs[i]
                    s, e = c["start"], c["end"]
                    print(f"        polsino_{prefix}_raw = np.array([{s}, {e}]) / 1000.0")
                    print(f"        self.polsino_{prefix}_fitto = self.prendi_punti_intermedi(polsino_{prefix}_raw[0], polsino_{prefix}_raw[1])\n")

        # 2. Hem (Fondo Maglia)
        hems = coordinates.get("Hem", [])
        if hems:
            h = hems[0]
            print("        self.fondo_maglia = np.array([")
            for p in h.get("points", []):
                print(f"            {p},")
            print("        ]) / 1000.0\n")

        # 3. Collar (Colletto)
        collars = coordinates.get("Collar", [])
        if collars:
            c = collars[0]
            print("        self.colletto = np.array([")
            for p in c.get("curve_points", []):
                print(f"            {p},")
            print("        ]) / 1000.0")
    else:
        print("\nNo objects detected. Try lowering conf_threshold or check model path.")

if __name__ == "__main__":
    test_unity()
