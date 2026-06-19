import cv2
import json
from yolo_inference import SweaterDetector
from geometry_utils import visualize_coordinates_3d

def main():
    # 1. Setup paths
    # Usiamo un'immagine dal dataset rgbd
    RGB_PATH = "dataset_rgbd_maglioncino\\20260619_101605_163183_rgb.png"
    DEPTH_PATH = "dataset_rgbd_maglioncino\\20260619_101605_163183_depth.png"
    MODEL_PATH = "sweater_segmentation\\yolo_run\\weights\\best.pt"
    OUTPUT_JSON = "robot_coordinates_3d.json"

    print(f"--- Inizio Verifica ---")
    print(f"RGB: {RGB_PATH}")
    print(f"Depth: {DEPTH_PATH}")

    # 2. Inizializza il detettore
    detector = SweaterDetector(MODEL_PATH)

    # 3. Processa l'immagine con depth
    print("\nEsecuzione inferenza YOLO...")
    coordinates, vis_img = detector.process_image(RGB_PATH, depth_path=DEPTH_PATH, conf_threshold=0.25)

    if not coordinates:
        print("Nessun oggetto rilevato. Verifica i percorsi o la soglia di confidenza.")
        return

    # 4. Salva i risultati
    detector.save_coordinates_json(coordinates, OUTPUT_JSON)
    cv2.imwrite("inference_result_depth.jpg", vis_img)
    print(f"Coordinate salvate in {OUTPUT_JSON}")
    print("Visualizzazione 2D salvata in inference_result_depth.jpg")

    # 5. Visualizza in 3D
    print("\nLancio visualizzazione 3D...")
    visualize_coordinates_3d(RGB_PATH, DEPTH_PATH, OUTPUT_JSON)

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
