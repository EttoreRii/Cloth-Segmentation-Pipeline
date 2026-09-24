import argparse
import cv2
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test trained YOLO model on a single image.")
    parser.add_argument(
        "--image", 
        type=str, 
        default=r"dataset_rgbd_maglioncino/20260922_103620_946363_rgb.png",
        help="Path to the image file to test."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=r"sweater_segmentation/yolo_run/weights/best.pt", 
        help="Path to the trained YOLO model weights."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold for inference."
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
    for result in results:
        res_plotted = result.plot()
        output_path = "inference_result.jpg"
        cv2.imwrite(output_path, res_plotted)
        print(f"Result saved to {output_path}")
        
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
