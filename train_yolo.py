import os
from ultralytics import YOLO

def train_yolo_local():
    """
    Local Laptop Training Script for YOLOv8 Object Detection & Classification.
    """
    print("[INFO] Starting Local Laptop Training for YOLOv8...")
    
    # Load pre-trained YOLOv8 Nano Segmentation model
    model = YOLO('yolov8n-seg.pt')
    
    # Check dataset location
    dataset_yaml = 'Mango-Roboflow-Dataset/data.yaml'
    if not os.path.exists(dataset_yaml):
        print(f"[WARNING] Dataset file '{dataset_yaml}' not found locally.")
        print("[INFO] Please download the YOLOv8 Instance Segmentation ZIP from Roboflow and unzip it as 'Mango-Roboflow-Dataset' in your project folder.")
        return

    # Train on Laptop CPU/GPU
    results = model.train(
        data=dataset_yaml,
        epochs=15,
        imgsz=640,
        batch=8,
        name='mango_seg_run'
    )
    
    # Rename best.pt to best_seg.pt for clarity
    trained_weights_path = 'runs/detect/mango_seg_run/weights/best.pt'
    target_weights_path = 'best_seg.pt'
    if os.path.exists(trained_weights_path):
        import shutil
        shutil.copy(trained_weights_path, target_weights_path)
        print(f"[INFO] Copied trained model weights to '{target_weights_path}'.")
    
    print("[SUCCESS] Local Laptop Training Completed!")

if __name__ == '__main__':
    train_yolo_local()
