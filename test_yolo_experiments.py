import os
import cv2
from ultralytics import YOLO

# Suppress Ultralytics verbose logging globally
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def run_tests():
    images = ['test_images/1.jpeg', 'test_images/2.jpeg', 'test_images/3.jpeg', 'test_images/apple.jpg']
    models = {
        'Segmentation (best_seg.pt)': 'best_seg.pt',
        'Detection (best.pt)': 'best.pt'
    }
    
    thresholds = [0.25, 0.20, 0.15, 0.10]
    img_sizes = [640, 1024]
    
    print("="*60)
    print("YOLO DETECTION EXPERIMENTS")
    print("="*60)
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}, file not found.")
            continue
            
        print(f"\\n>>> LOADING {model_name} <<<")
        model = YOLO(model_path)
        
        for img_path in images:
            if not os.path.exists(img_path):
                print(f"  Missing {img_path}")
                continue
                
            print(f"\\n--- IMAGE: {os.path.basename(img_path)} ---")
            img = cv2.imread(img_path)
            
            # Test sizes
            for size in img_sizes:
                print(f"  [Image Size: {size}]")
                # Test Thresholds
                for conf in thresholds:
                    # Run YOLO
                    results = model(img, conf=conf, imgsz=size, verbose=False)
                    boxes = results[0].boxes
                    
                    if len(boxes) > 0:
                        top_box = boxes[0]
                        conf_val = float(top_box.conf[0])
                        cls_val = int(top_box.cls[0])
                        xyxy = top_box.xyxy[0].cpu().numpy().astype(int)
                        has_mask = hasattr(results[0], 'masks') and results[0].masks is not None
                        
                        print(f"    Conf {conf:.2f}: Det={len(boxes)} | BestConf={conf_val:.4f} | Cls={cls_val} | Box={xyxy.tolist()} | Mask={has_mask}")
                    else:
                        print(f"    Conf {conf:.2f}: NO DETECTIONS")

if __name__ == '__main__':
    run_tests()
