import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def get_data_yaml():
    candidates = [
        os.path.join(PROJECT_ROOT, "data.yaml"),
        os.path.join(PROJECT_ROOT, "scripts + immagini x train", "data.yaml"),
        os.path.join(SCRIPT_DIR, "data.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "data.yaml"

def get_base_model():
    candidates = [
        os.path.join(PROJECT_ROOT, "yolov8n-seg.pt"),
        os.path.join(SCRIPT_DIR, "yolov8n-seg.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolov8n-seg.pt"

def train_yolo(epochs=30, batch=4, device='cpu'):
    # Load base model (YOLOv8 Nano Segmentation model)
    model_path = get_base_model()
    data_path = get_data_yaml()

    print(f"Loading model: {model_path}")
    print(f"Using dataset config: {data_path}")
    model = YOLO(model_path) 

    # Output project directory
    project_dir = os.path.join(PROJECT_ROOT, "risultati train-val")

    # Train the model
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        imgsz=640, 
        batch=batch,
        device=device,
        project=project_dir,
        name="yolo_run"
    )
    
    # Export the model to ONNX
    success = model.export(format="onnx") 
    print("Training complete. Exported:", success)

if __name__ == "__main__":
    train_yolo()

