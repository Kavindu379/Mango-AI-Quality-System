import os
from ultralytics import YOLO

def train_yolo_local():
    """
    Local Laptop Training Script for YOLOv8 Object Detection & Classification.
    """
    print("[INFO] Starting Local Laptop Training for YOLOv8...")
    
    # Load pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt')
    
    # Check dataset location
    dataset_yaml = 'Mango-Roboflow-Dataset/data.yaml'
    if not os.path.exists(dataset_yaml):
        print(f"[WARNING] Dataset file '{dataset_yaml}' not found locally.")
        print("[INFO] Please download the YOLOv8 ZIP from Roboflow and unzip it as 'Mango-Roboflow-Dataset' in your project folder.")
        return

    # Train on Laptop CPU/GPU
    results = model.train(
        data=dataset_yaml,
        epochs=15,
        imgsz=640,
        batch=8,
        name='mango_local_run'
    )
    
    print("[SUCCESS] Local Laptop Training Completed!")
    print(f"[INFO] Saved trained model weights to 'runs/detect/mango_local_run/weights/best.pt'.")

if __name__ == '__main__':
    train_yolo_local()
