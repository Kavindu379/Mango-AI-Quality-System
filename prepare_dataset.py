import os
import shutil
import random
from PIL import Image
import numpy as np
from ultralytics import YOLO

def process_and_split_dataset():
    random.seed(42)
    source_dirs = ['dataset/train', 'dataset/val']
    dest_dir = 'dataset_cropped'
    
    classes = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe', 'Non_Mango']
    
    # Collect all images
    all_images = {cls: [] for cls in classes}
    for sdir in source_dirs:
        for cls in classes:
            cpath = os.path.join(sdir, cls)
            if os.path.exists(cpath):
                for f in os.listdir(cpath):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        all_images[cls].append(os.path.join(cpath, f))
                        
    # Load YOLO model
    yolo_path = 'best_seg.pt' if os.path.exists('best_seg.pt') else 'best.pt'
    try:
        yolo_model = YOLO(yolo_path)
        print(f"Loaded YOLO model: {yolo_path}")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    # Create dest directories
    splits = ['train', 'val', 'test']
    for sp in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, sp, cls), exist_ok=True)
            
    stats = {
        'total': 0,
        'cropped': 0,
        'failed': 0,
        'split_counts': {'train': 0, 'val': 0, 'test': 0},
        'class_counts': {cls: 0 for cls in classes}
    }
    
    for cls in classes:
        images = all_images[cls]
        # Sort for deterministic shuffling
        images.sort()
        random.shuffle(images)
        
        n_total = len(images)
        if n_total == 0:
            continue
            
        n_train = int(n_total * 0.75)
        n_val = int(n_total * 0.15)
        # test gets the rest
        
        for idx, img_path in enumerate(images):
            if idx < n_train:
                split = 'train'
            elif idx < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
                
            fname = os.path.basename(img_path)
            dest_path = os.path.join(dest_dir, split, cls, f"{idx}_{fname}")
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                
                # YOLO detection
                results = yolo_model(img_np, conf=0.25, verbose=False)
                cropped = False
                
                if len(results[0].boxes) > 0:
                    top_box = results[0].boxes[0]
                    xyxy = top_box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, xyxy[0]), max(0, xyxy[1])
                    x2, y2 = min(img_np.shape[1], xyxy[2]), min(img_np.shape[0], xyxy[3])
                    
                    if (x2 - x1) > 20 and (y2 - y1) > 20:
                        crop_img = img_np[y1:y2, x1:x2]
                        final_img = Image.fromarray(crop_img).resize((224, 224))
                        cropped = True
                        stats['cropped'] += 1
                
                if not cropped:
                    # Fallback to original image resized
                    final_img = img.resize((224, 224))
                    stats['failed'] += 1
                    print(f"YOLO failed to detect object in {img_path}. Using resized original.")
                    
                final_img.save(dest_path)
                
                stats['total'] += 1
                stats['split_counts'][split] += 1
                stats['class_counts'][cls] += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    print("\n" + "="*50)
    print("DATASET PREPARATION REPORT")
    print("="*50)
    print(f"Total Images Processed: {stats['total']}")
    print(f"Successfully Cropped:   {stats['cropped']}")
    print(f"Failed Detections:      {stats['failed']}")
    print("\nSplit Distribution:")
    for sp in splits:
        print(f"  {sp}: {stats['split_counts'][sp]}")
    print("\nClass Distribution:")
    for cls in classes:
        print(f"  {cls}: {stats['class_counts'][cls]}")
    print("="*50)
    print(f"Cropped dataset is ready at '{dest_dir}/'.")

if __name__ == '__main__':
    process_and_split_dataset()
