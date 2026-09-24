import cv2
import json
import os
from yolo_inference import SweaterDetector, resolve_model_path, PROJECT_ROOT
from geometry_utils import visualize_coordinates_3d

def main():
    # 1. Setup paths
    # Usiamo un'immagine dal dataset rgbd
    default_rgb = os.path.join(PROJECT_ROOT, "dataset_rgbd_maglioncino", "20260109_153911_938455_rgb.png")
    default_depth = os.path.join(PROJECT_ROOT, "dataset_rgbd_maglioncino", "20260109_153911_938455_depth.png")
    
    if not (os.path.exists(default_rgb) and os.path.exists(default_depth)):
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset_rgbd_maglioncino")
        if os.path.exists(dataset_dir):
            for f in os.listdir(dataset_dir):
                if f.endswith("_rgb.png"):
                    prefix = f[:-8]
                    cand_depth = os.path.join(dataset_dir, f"{prefix}_depth.png")
                    if os.path.exists(cand_depth):
                        default_rgb = os.path.join(dataset_dir, f)
                        default_depth = cand_depth
                        break

    RGB_PATH = default_rgb
    DEPTH_PATH = default_depth
    MODEL_PATH = resolve_model_path()
    
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "risultati rete")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    OUTPUT_JSON = os.path.join(RESULTS_DIR, "robot_coordinates_3d.json")
    OUTPUT_IMG = os.path.join(RESULTS_DIR, "inference_result_depth.jpg")

    print(f"--- Inizio Verifica ---")
    print(f"RGB: {RGB_PATH}")
    print(f"Depth: {DEPTH_PATH}")

    # 2. Inizializza il detettore con intrinseci corrispondenti alla risoluzione
    test_img = cv2.imread(RGB_PATH)
    if test_img is not None and test_img.shape[1] == 1280:
        detector = SweaterDetector(MODEL_PATH, fx=908.89948, fy=908.29236, ppx=647.69177, ppy=356.04810)
    else:
        # Default 640x480
        detector = SweaterDetector(MODEL_PATH, fx=605.93298, fy=605.52826, ppx=325.12787, ppy=237.36540)

    # 3. Processa l'immagine con depth
    print("\nEsecuzione inferenza YOLO...")
    coordinates, vis_img = detector.process_image(RGB_PATH, depth_path=DEPTH_PATH, conf_threshold=0.25)

    if not coordinates:
        print("Nessun oggetto rilevato. Verifica i percorsi o la soglia di confidenza.")
        return

    # 4. Salva i risultati in 'risultati rete'
    detector.save_coordinates_json(coordinates, OUTPUT_JSON)
    cv2.imwrite(OUTPUT_IMG, vis_img)
    print(f"Coordinate salvate in {OUTPUT_JSON}")
    print(f"Visualizzazione 2D salvata in {OUTPUT_IMG}")

    # 5. Visualizza in 3D
    print("\nLancio visualizzazione 3D...")
    try:
        visualize_coordinates_3d(RGB_PATH, DEPTH_PATH, OUTPUT_JSON)
    except Exception as e:
        print(f"Avviso visualizzazione 3D: {e}")

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
                print(f"        polsino_{prefix}_raw = self.rete_to_base(np.array([{s}, {e}]))")
                print(f"        self.polsino_{prefix}_fitto = self.prendi_punti_intermedi(polsino_{prefix}_raw[0], polsino_{prefix}_raw[1])\n")

    # 2. Hem (Fondo Maglia)
    hems = coordinates.get("Hem", [])
    if hems:
        h = hems[0]
        print("        self.fondo_maglia = self.rete_to_base(np.array([")
        for p in h.get("points", []):
            print(f"            {p},")
        print("        ]))\n")

    # 3. Collar (Colletto)
    collars = coordinates.get("Collar", [])
    if collars:
        c = collars[0]
        print("        self.colletto = self.rete_to_base(np.array([")
        for p in c.get("curve_points", []):
            print(f"            {p},")
        print("        ]))")
    else:
        print("\nNo objects detected. Try lowering conf_threshold or check model path.")

if __name__ == "__main__":
    main()
