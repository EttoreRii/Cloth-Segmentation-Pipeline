from ultralytics import YOLO

def train_yolo():
    # Load a model
    # "yolov8n-seg.pt" is the Nano Segmentation model (fastest, lightweight)
    # ideal for Jetson Orin Nano (8GB)
    model = YOLO("yolov8n-seg.pt") 

    # Train the model (CPU Mode)
    # batch=4 is safer for CPU. epochs=20 is a quick test.
    results = model.train(
        data="data.yaml", 
        epochs=30, 
        imgsz=640, 
        batch=4,
        device='cpu', # Force CPU
        project="sweater_segmentation",
        name="yolo_run"
    )
    
    # Export the model to ONNX (Portable, runs everywhere)
    # TensorRT (.engine) is best done ON THE DEVICE (Jetson) to avoid version mismatches.
    success = model.export(format="onnx") 
    print("Training complete. Exported:", success)

if __name__ == "__main__":
    train_yolo()
