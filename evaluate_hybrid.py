import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from ultralytics import YOLO
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from model import build_mango_cnn_model, CLASS_NAMES
from preprocess import preprocess_image_pytorch, extract_hsv_color_analysis, validate_is_mango_candidate
from hybrid_ripeness import calculate_hybrid_scores

def get_models():
    # Load YOLO
    yolo_path = 'best_seg.pt' if os.path.exists('best_seg.pt') else 'best.pt'
    yolo_model = YOLO(yolo_path)
    
    # Load PyTorch CNN
    p_m = build_mango_cnn_model(num_classes=len(CLASS_NAMES), model_type='efficientnet')
    p_m.load_state_dict(torch.load('mango_model.pth', map_location='cpu'), strict=False)
    p_m.eval()
    return yolo_model, p_m

def main():
    print("="*60)
    print("WARNING: A proper holdout 'test' set does not exist!")
    print("Evaluating on the 'val' set as a substitute for testing.")
    print("Do NOT use these metrics for final scientific publication.")
    print("="*60)
    
    test_dir = 'dataset/val'
    if not os.path.exists(test_dir):
        print(f"Error: Could not find {test_dir}. Cannot evaluate.")
        return

    y_m, p_m = get_models()
    results = []
    
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            
            img_path = os.path.join(cls_dir, fname)
            image = Image.open(img_path).convert('RGB')
            img_rgb = np.array(image)
            
            cropped_mango_rgb = img_rgb
            has_yolo_detection = False
            mango_mask_np = None
            
            # YOLO Pipeline
            y_results = y_m(img_rgb, conf=0.25, verbose=False)
            if len(y_results[0].boxes) > 0:
                top_box = y_results[0].boxes[0]
                xyxy = top_box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(img_rgb.shape[1], xyxy[2]), min(img_rgb.shape[0], xyxy[3])
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    cropped_mango_rgb = img_rgb[y1:y2, x1:x2]
                    
                    if hasattr(y_results[0], 'masks') and y_results[0].masks is not None and len(y_results[0].masks) > 0:
                        import cv2
                        full_mask = y_results[0].masks.data[0].cpu().numpy()
                        full_mask_resized = cv2.resize(full_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mango_mask_np = full_mask_resized[y1:y2, x1:x2]
                        mango_mask_np = (mango_mask_np > 0.5).astype(np.uint8) * 255
                has_yolo_detection = True
                
            color_features = extract_hsv_color_analysis(cropped_mango_rgb, mask_np=mango_mask_np)
            
            # CNN Pipeline
            cropped_pil = Image.fromarray(cropped_mango_rgb)
            cropped_tensor, _ = preprocess_image_pytorch(cropped_pil)
            with torch.no_grad():
                # Apply TTA (Test Time Augmentation) as used in server.py
                logits_orig = p_m(cropped_tensor)
                probs_orig = F.softmax(logits_orig, dim=1).numpy()[0]
                
                flipped_tensor = torch.flip(cropped_tensor, dims=[3])
                logits_flip = p_m(flipped_tensor)
                probs_flip = F.softmax(logits_flip, dim=1).numpy()[0]
                
                probs = (probs_orig + probs_flip) / 2.0
                
                pred_idx = int(np.argmax(probs))
                pred_class = CLASS_NAMES[pred_idx]
                conf = float(probs[pred_idx])
                
                class_probs = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
                
            # OOD Pipeline
            is_valid_mango, _ = validate_is_mango_candidate(
                color_features, conf, cropped_mango_rgb, has_yolo_detection=has_yolo_detection, mask_np=mango_mask_np
            )
            
            cnn_final_class = pred_class
            if not is_valid_mango or pred_class == 'Non_Mango':
                cnn_final_class = 'Non_Mango'
                class_probs = {k: 0.0 for k in class_probs}
                class_probs['Non_Mango'] = 100.0
                is_valid_mango = False

            # Hybrid Pipeline
            # Note: We temporarily suppress the print statements from hybrid_ripeness
            import sys
            import io
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            hybrid_probs, hybrid_pred, hybrid_conf, is_uncertain = calculate_hybrid_scores(
                class_probs, color_features, is_valid_mango
            )
            sys.stdout = original_stdout
            
            results.append({
                'filename': fname,
                'ground_truth': cls_name,
                'cnn_pred': cnn_final_class,
                'cnn_conf': conf * 100,
                'hybrid_pred': hybrid_pred,
                'hybrid_conf': hybrid_conf,
                'yellow_pct': color_features.get('yellow_percentage', 0.0),
                'green_pct': color_features.get('green_percentage', 0.0),
                'dark_pct': color_features.get('dark_spots_percentage', 0.0),
                'cnn_correct': cnn_final_class == cls_name,
                'hybrid_correct': hybrid_pred == cls_name
            })
            
    if not results:
        print("No images found in the dataset.")
        return

    df = pd.DataFrame(results)
    
    y_true = df['ground_truth']
    y_cnn = df['cnn_pred']
    y_hyb = df['hybrid_pred']
    
    cnn_acc = accuracy_score(y_true, y_cnn)
    hyb_acc = accuracy_score(y_true, y_hyb)
    
    cnn_p, cnn_r, cnn_f, _ = precision_recall_fscore_support(y_true, y_cnn, average='weighted', zero_division=0)
    hyb_p, hyb_r, hyb_f, _ = precision_recall_fscore_support(y_true, y_hyb, average='weighted', zero_division=0)
    
    print("\n--- PERFORMANCE COMPARISON ---")
    print(f"{'Metric':<10} | {'EfficientNet-B0':<15} | {'Hybrid':<15}")
    print("-" * 45)
    print(f"{'Accuracy':<10} | {cnn_acc:.4f}          | {hyb_acc:.4f}")
    print(f"{'Precision':<10} | {cnn_p:.4f}          | {hyb_p:.4f}")
    print(f"{'Recall':<10} | {cnn_r:.4f}          | {hyb_r:.4f}")
    print(f"{'F1':<10} | {cnn_f:.4f}          | {hyb_f:.4f}")
    
    print("\n--- CONFUSION MATRICES ---")
    print("Classes:", CLASS_NAMES)
    print("\nCNN Confusion Matrix:")
    print(confusion_matrix(y_true, y_cnn, labels=CLASS_NAMES))
    print("\nHybrid Confusion Matrix:")
    print(confusion_matrix(y_true, y_hyb, labels=CLASS_NAMES))
    
    print("\n--- ANALYSIS OF TRANSITIONS ---")
    cnn_right_hyb_wrong = df[(df['cnn_correct'] == True) & (df['hybrid_correct'] == False)]
    print(f"\nCases where CNN was Correct but Hybrid got it Wrong: {len(cnn_right_hyb_wrong)}")
    for _, row in cnn_right_hyb_wrong.iterrows():
        print(f"File: {row['filename']} | GT: {row['ground_truth']} | CNN: {row['cnn_pred']} ({row['cnn_conf']:.1f}%) | HYB: {row['hybrid_pred']} ({row['hybrid_conf']:.1f}%) | Y: {row['yellow_pct']}%, G: {row['green_pct']}%, D: {row['dark_pct']}%")

    cnn_wrong_hyb_right = df[(df['cnn_correct'] == False) & (df['hybrid_correct'] == True)]
    print(f"\nCases where CNN was Wrong but Hybrid got it Right: {len(cnn_wrong_hyb_right)}")
    for _, row in cnn_wrong_hyb_right.iterrows():
        print(f"File: {row['filename']} | GT: {row['ground_truth']} | CNN: {row['cnn_pred']} ({row['cnn_conf']:.1f}%) | HYB: {row['hybrid_pred']} ({row['hybrid_conf']:.1f}%) | Y: {row['yellow_pct']}%, G: {row['green_pct']}%, D: {row['dark_pct']}%")

if __name__ == '__main__':
    main()
