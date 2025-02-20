from ultralytics import YOLO
import torch
import sys
import os
def train_yolo(data_path, model_path, project="runs", name="exp", epochs=100, batch_size=16, img_size=640, learning_rate=0.001):
    print(f"""
    Starting YOLO training with parameters:
    - Project: {project}
    - Name: {name}
    - Epochs: {epochs}
    - Batch size: {batch_size}
    - Image size: {img_size}
    - Learning rate: {learning_rate}
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # project 경로 앞에 'logs' 추가
    project_path = os.path.join("logs", project)
    
    model = YOLO(model_path)
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        device=device,
        project=project_path,  # 수정된 project 경로 사용
        name=name,
        exist_ok=True
    )
    print("Training completed.")

def train_ocr(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):

    pass

def train_custom_model(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):

    pass