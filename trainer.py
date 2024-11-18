from ultralytics import YOLO
import sys

def train_yolo(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):
    
    print(f"Training started for dataset: {dataset_path} with model: {model_path}")
    
    model = YOLO(model_path)
    model.train(
        data=dataset_path, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch, 
        device=device,
        project="runs",
        name="train",
        exist_ok=True
    )

    print("Training completed.")

def train_ocr(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):

    pass

def train_custom_model(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):

    pass