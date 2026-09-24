import argparse
import cv2
import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def get_default_model():
    candidates = [
        os.path.join(PROJECT_ROOT, "risultati train-val", "yolo_run", "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "sweater_segmentation_best", "yolo_run", "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "yolov8n-seg.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def main():
    default_model = get_default_model()
    default_img = os.path.join(PROJECT_ROOT, "dataset_rgbd_maglioncino", "20260109_153431_498491_rgb.png")
    default_out = os.path.join(PROJECT_ROOT, "risultati rete", "inference_result.jpg")

    parser = argparse.ArgumentParser(description="Test trained YOLO model on a single image.")
    parser.add_argument(
        "--image", 
        type=str, 
        default=default_img,
        help="Path to the image file to test."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=default_model, 
        help="Path to the trained YOLO model weights."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold for inference."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_out,
        help="Path to save the plotted inference result."
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"Running inference on: {args.image} with conf={args.conf}")
    results = model(args.image, conf=args.conf)
    
    # Visualize the results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for result in results:
        res_plotted = result.plot()
        cv2.imwrite(args.output, res_plotted)
        print(f"Result saved to {args.output}")
        
        # Try to show the image if possible (might not work in all environments)
        try:
            cv2.imshow("Inference Result", res_plotted)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Could not display image (GUI might not be available). Check the saved file.")

if __name__ == "__main__":
    main()
