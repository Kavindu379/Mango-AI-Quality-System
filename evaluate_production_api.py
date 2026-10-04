import os
import requests
import json
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

API_URL = "http://localhost:5000/api/predict"

CLASS_NAMES = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe', 'Non_Mango']
CLASS_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}

def run_evaluation():
    val_dir = 'dataset/val'
    
    y_true = []
    y_pred = []
    
    mango_true = []
    mango_pred = []
    
    results_log = []
    
    total_images = sum([len(files) for r, d, files in os.walk(val_dir)])
    processed = 0
    
    print("="*60)
    print("PRODUCTION PIPELINE CNN EVALUATION")
    print("="*60)
    
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        for img_path in glob.glob(os.path.join(cls_dir, "*.*")):
            with open(img_path, 'rb') as f:
                try:
                    resp = requests.post(API_URL, files={'image': f})
                    data = resp.json()
                except Exception as e:
                    print(f"Error on {img_path}: {e}")
                    continue
            
            processed += 1
            filename = os.path.basename(img_path)
            gt_label = cls
            
            if data.get('success'):
                cnn_probs = data['prediction']['class_probabilities']
                
                # Determine raw CNN prediction
                best_cls = max(cnn_probs, key=cnn_probs.get)
                best_conf = cnn_probs[best_cls]
                
                y_true.append(CLASS_MAP[gt_label])
                y_pred.append(CLASS_MAP[best_cls])
                
                if gt_label != 'Non_Mango':
                    mango_true.append(CLASS_MAP[gt_label])
                    mango_pred.append(CLASS_MAP[best_cls])
                
                # Only log errors or specific cases (1.jpeg) to keep output clean
                is_correct = (gt_label == best_cls)
                if filename == '1.jpeg' or not is_correct:
                    print(f"[{filename}] True: {gt_label} | Pred: {best_cls} ({best_conf}%) | Correct: {is_correct}")
                    if filename == '1.jpeg':
                        print(f"   => 1.jpeg Detailed Probs: {cnn_probs}")
            else:
                # YOLO failed
                # If Unable_To_Detect, we treat it as CNN prediction Non_Mango? Or just mark it as failure?
                # For strict accuracy against Colab (where Colab used YOLO crops but if YOLO failed it might have been dropped or failed),
                # If YOLO fails, let's treat it as Non_Mango.
                best_cls = 'Non_Mango'
                y_true.append(CLASS_MAP[gt_label])
                y_pred.append(CLASS_MAP[best_cls])
                
                if gt_label != 'Non_Mango':
                    mango_true.append(CLASS_MAP[gt_label])
                    mango_pred.append(CLASS_MAP[best_cls])
                    
                print(f"[{filename}] True: {gt_label} | Pred: {best_cls} (YOLO FAILED) | Correct: {gt_label == best_cls}")
                if filename == '1.jpeg':
                    print(f"   => 1.jpeg YOLO FAILED")

    print(f"\\nProcessed {processed}/{total_images} images.\\n")
    
    if not y_true:
        print("No images processed.")
        return
        
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2,3], zero_division=0)
    
    print(f"Production CNN Accuracy: {acc*100:.2f}%")
    if mango_true:
        mango_acc = accuracy_score(mango_true, mango_pred)
        print(f"Mango-only Accuracy: {mango_acc*100:.2f}%")
        
    print("\\nPer-class Metrics:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:18s}: Precision: {p[i]*100:.2f}%, Recall: {r[i]*100:.2f}%, F1: {f[i]*100:.2f}%")
        
    print("\\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0,1,2,3]))

if __name__ == '__main__':
    run_evaluation()
