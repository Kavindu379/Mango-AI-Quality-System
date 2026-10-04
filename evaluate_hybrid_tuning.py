import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

from preprocess import preprocess_image_pytorch, extract_hsv_color_analysis, validate_is_mango_candidate
from model import build_mango_cnn_model, CLASS_NAMES
from hybrid_ripeness import calculate_hybrid_scores

# Setup models
YOLO_MODEL_PATH = 'best_seg.pt'
PYTORCH_MODEL_PATH = 'mango_model.pth'

yolo_model = YOLO(YOLO_MODEL_PATH)
pytorch_model = build_mango_cnn_model(num_classes=4, model_type='efficientnet')
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
pytorch_model.eval()

VAL_DIR = 'dataset/val'
CACHE_FILE = 'validation_features_cache.json'
CLASS_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}

def extract_features(img_path):
    image = Image.open(img_path).convert('RGB')
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    results = yolo_model(img_bgr, conf=0.25, imgsz=640, verbose=False)
    has_yolo_detection = False
    cropped_mango_rgb = img_rgb
    mango_mask_np = None
    
    if len(results[0].boxes) > 0:
        top_box = results[0].boxes[0]
        xyxy = top_box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(img_rgb.shape[1], xyxy[2]), min(img_rgb.shape[0], xyxy[3])
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            cropped_mango_rgb = img_rgb[y1:y2, x1:x2]
            if hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks) > 0:
                full_mask = results[0].masks.data[0].cpu().numpy()
                full_mask_resized = cv2.resize(full_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                mango_mask_np = full_mask_resized[y1:y2, x1:x2]
                mango_mask_np = (mango_mask_np > 0.5).astype(np.uint8) * 255
        has_yolo_detection = True
    else:
        # If YOLO fails entirely, we record that it failed so we can properly evaluate
        return {'success': False, 'error': 'YOLO FAILED'}

    color_features = extract_hsv_color_analysis(cropped_mango_rgb, mask_np=mango_mask_np)
    
    cropped_pil = Image.fromarray(cropped_mango_rgb)
    cropped_tensor, _ = preprocess_image_pytorch(cropped_pil)
    
    with torch.no_grad():
        logits_orig = pytorch_model(cropped_tensor)
        probs_orig = F.softmax(logits_orig, dim=1).numpy()[0]
        flipped_tensor = torch.flip(cropped_tensor, dims=[3])
        logits_flip = pytorch_model(flipped_tensor)
        probs_flip = F.softmax(logits_flip, dim=1).numpy()[0]
        probabilities = (probs_orig + probs_flip) / 2.0
        
        pred_index = int(np.argmax(probabilities))
        pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else 'Non_Mango'
        conf = float(probabilities[pred_index])
        
        class_probs = {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

    is_valid_mango, validation_msg = validate_is_mango_candidate(
        color_features, conf, cropped_mango_rgb, has_yolo_detection=has_yolo_detection, mask_np=mango_mask_np
    )
    
    return {
        'success': True,
        'class_probs': class_probs,
        'pred_class': pred_class,
        'color_features': color_features,
        'is_valid_mango': is_valid_mango
    }

def get_or_build_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
            
    cache = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(VAL_DIR, cls)
        if not os.path.exists(cls_dir): continue
        for img_path in glob.glob(os.path.join(cls_dir, "*.*")):
            print(f"Extracting {img_path}...")
            feats = extract_features(img_path)
            cache[img_path] = {'ground_truth': cls, 'features': feats}
            
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    return cache

def calculate_metrics(y_true, y_pred, mango_only=False):
    if not y_true: return {}
    
    if mango_only:
        filtered_true, filtered_pred = [], []
        for t, p in zip(y_true, y_pred):
            if t != 'Non_Mango':
                filtered_true.append(t)
                filtered_pred.append(p)
        y_true, y_pred = filtered_true, filtered_pred
        labels = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe']
    else:
        labels = CLASS_NAMES

    if not y_true: return {}

    y_true_idx = [labels.index(x) if x in labels else -1 for x in y_true]
    y_pred_idx = [labels.index(x) if x in labels else -1 for x in y_pred]

    acc = accuracy_score(y_true_idx, y_pred_idx)
    p, r, f, _ = precision_recall_fscore_support(y_true_idx, y_pred_idx, labels=range(len(labels)), average='macro', zero_division=0)
    w_p, w_r, w_f, _ = precision_recall_fscore_support(y_true_idx, y_pred_idx, labels=range(len(labels)), average='weighted', zero_division=0)
    
    return {
        'accuracy': acc, 'macro_precision': p, 'macro_recall': r, 'macro_f1': f, 'weighted_f1': w_f
    }

def run_grid_search():
    cache = get_or_build_cache()
    
    results = []
    
    cnn_weights = [0.70, 0.75, 0.80, 0.85, 0.90]
    hsv_weights = [0.05, 0.10, 0.15, 0.20]
    defect_thresholds = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    uncertainty_margins = [0.05, 0.07, 0.10, 0.15]
    
    print(f"Running grid search over {len(cnn_weights)*len(hsv_weights)*len(defect_thresholds)*len(uncertainty_margins)} configurations...")
    
    # Calculate BASELINE CNN metrics
    base_true = []
    base_pred = []
    for img_path, data in cache.items():
        gt = data['ground_truth']
        feats = data['features']
        base_true.append(gt)
        if not feats['success']:
            base_pred.append('Non_Mango')
        else:
            # Baseline is just CNN prediction
            base_pred.append(feats['pred_class'])
            
    base_metrics = calculate_metrics(base_true, base_pred)
    base_mango_metrics = calculate_metrics(base_true, base_pred, mango_only=True)
    print(f"--- BASELINE CNN ---")
    print(f"Accuracy: {base_metrics['accuracy']:.4f}, Mango-only F1: {base_mango_metrics['macro_f1']:.4f}")
    print("--------------------")

    for cnn_w in cnn_weights:
        for hsv_w in hsv_weights:
            defect_w = round(1.0 - cnn_w - hsv_w, 2)
            if defect_w < 0.0 or defect_w > 0.30: continue
            
            for d_thresh in defect_thresholds:
                for u_margin in uncertainty_margins:
                    
                    y_true = []
                    y_pred = []
                    uncertain_count = 0
                    
                    cnn_correct_hybrid_wrong = 0
                    cnn_wrong_hybrid_correct = 0
                    
                    for img_path, data in cache.items():
                        gt = data['ground_truth']
                        feats = data['features']
                        
                        y_true.append(gt)
                        
                        if not feats['success']:
                            y_pred.append('Non_Mango')
                            continue
                            
                        # Run parameterized hybrid logic
                        class_probs = feats['class_probs']
                        
                        # 1. Non_Mango Override Logic (from server.py)
                        # We must preserve Non_Mango safety!
                        is_valid = feats['is_valid_mango']
                        cnn_pred = feats['pred_class']
                        
                        if not is_valid or cnn_pred == 'Non_Mango':
                            hybrid_pred = 'Non_Mango'
                        else:
                            # 2. Run Hybrid Fusion
                            _, hybrid_pred, _, is_unc = calculate_hybrid_scores(
                                class_probs, feats['color_features'], is_valid,
                                cnn_w, hsv_w, defect_w, d_thresh, u_margin
                            )
                            if is_unc: uncertain_count += 1
                            
                        y_pred.append(hybrid_pred)
                        
                        # Compare CNN baseline vs Hybrid
                        if cnn_pred == gt and hybrid_pred != gt:
                            cnn_correct_hybrid_wrong += 1
                        if cnn_pred != gt and hybrid_pred == gt:
                            cnn_wrong_hybrid_correct += 1

                    m = calculate_metrics(y_true, y_pred)
                    m_mango = calculate_metrics(y_true, y_pred, mango_only=True)
                    
                    results.append({
                        'CNN_W': cnn_w,
                        'HSV_W': hsv_w,
                        'DEFECT_W': defect_w,
                        'D_THRESH': d_thresh,
                        'U_MARGIN': u_margin,
                        'Accuracy': m.get('accuracy', 0),
                        'Macro_Precision': m.get('macro_precision', 0),
                        'Macro_Recall': m.get('macro_recall', 0),
                        'Macro_F1': m.get('macro_f1', 0),
                        'Weighted_F1': m.get('weighted_f1', 0),
                        'Mango_Accuracy': m_mango.get('accuracy', 0),
                        'Mango_F1': m_mango.get('macro_f1', 0),
                        'CNN_Cor->Hyb_Wr': cnn_correct_hybrid_wrong,
                        'CNN_Wr->Hyb_Cor': cnn_wrong_hybrid_correct,
                        'Uncertainties': uncertain_count
                    })
                    
    df = pd.DataFrame(results)
    df.to_csv('hybrid_experiment_results.csv', index=False)
    
    # Sort primarily by Mango_F1, then Macro_F1, then minimizing regressions
    df_sorted = df.sort_values(by=['Mango_F1', 'Macro_F1', 'CNN_Cor->Hyb_Wr'], ascending=[False, False, True])
    
    print("\nTOP 5 CONFIGURATIONS:")
    print(df_sorted.head(5).to_string())

if __name__ == '__main__':
    run_grid_search()
