import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from preprocess import preprocess_image_pytorch, extract_hsv_color_analysis, validate_is_mango_candidate
from model import build_mango_cnn_model, CLASS_NAMES
from hybrid_ripeness import calculate_hybrid_scores

YOLO_MODEL_PATH = 'best_seg.pt'
PYTORCH_MODEL_PATH = 'mango_model.pth'

def get_test_images():
    random.seed(42)
    source_dirs = ['dataset/train', 'dataset/val']
    classes = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe', 'Non_Mango']
    
    all_images = {cls: [] for cls in classes}
    for sdir in source_dirs:
        for cls in classes:
            cpath = os.path.join(sdir, cls)
            if os.path.exists(cpath):
                for f in os.listdir(cpath):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        all_images[cls].append(os.path.join(cpath, f))
                        
    test_images = []
    for cls in classes:
        images = all_images[cls]
        images.sort()
        random.shuffle(images)
        n_total = len(images)
        if n_total == 0: continue
        n_train = int(n_total * 0.75)
        n_val = int(n_total * 0.15)
        
        for idx, img_path in enumerate(images):
            if idx >= n_train + n_val:
                test_images.append((img_path, cls))
    return test_images

def run_test():
    yolo_model = YOLO(YOLO_MODEL_PATH)
    pytorch_model = build_mango_cnn_model(num_classes=4, model_type='efficientnet')
    pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
    pytorch_model.eval()

    test_images = get_test_images()
    print(f"Testing on {len(test_images)} untouched original test images...")
    
    y_true = []
    y_cnn = []
    y_hybrid = []
    
    for img_path, cls in test_images:
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
        
        y_true.append(cls)
        if not has_yolo_detection:
            y_cnn.append('Non_Mango')
            y_hybrid.append('Non_Mango')
            continue
            
        color_features = extract_hsv_color_analysis(cropped_mango_rgb, mask_np=mango_mask_np)
        cropped_pil = Image.fromarray(cropped_mango_rgb)
        cropped_tensor, _ = preprocess_image_pytorch(cropped_pil)
        
        with torch.no_grad():
            logits_orig = pytorch_model(cropped_tensor)
            probs_orig = F.softmax(logits_orig, dim=1).numpy()[0]
            flipped_tensor = torch.flip(cropped_tensor, dims=[3])
            logits_flip = pytorch_model(flipped_tensor)
            probs_flip = F.softmax(logits_flip, dim=1).numpy()[0]
            probs = (probs_orig + probs_flip) / 2.0
            
            pred_index = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else 'Non_Mango'
            class_probs = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
        
        is_valid_mango, _ = validate_is_mango_candidate(color_features, probs[pred_index], cropped_mango_rgb, has_yolo_detection=True, mask_np=mango_mask_np)
        
        y_cnn.append(pred_class)
        
        if not is_valid_mango or pred_class == 'Non_Mango':
            hybrid_pred = 'Non_Mango'
        else:
            _, hybrid_pred, _, _ = calculate_hybrid_scores(
                class_probs, color_features, is_valid_mango,
                cnn_w=0.85, hsv_w=0.05, defect_w=0.10, defect_threshold=15.0, margin_threshold=0.10
            )
        y_hybrid.append(hybrid_pred)

    def calc(t, p):
        if not t: return 0,0,0
        t_m, p_m = zip(*[(x, y) for x, y in zip(t, p) if x != 'Non_Mango']) if any(x != 'Non_Mango' for x in t) else ([], [])
        acc = accuracy_score(t, p)
        mac_f1 = precision_recall_fscore_support(t, p, average='macro', zero_division=0)[2]
        m_acc = accuracy_score(t_m, p_m) if t_m else 0
        return acc, mac_f1, m_acc
        
    print(f"CNN    -> Acc: {calc(y_true, y_cnn)[0]:.4f} | Macro F1: {calc(y_true, y_cnn)[1]:.4f} | Mango Acc: {calc(y_true, y_cnn)[2]:.4f}")
    print(f"HYBRID -> Acc: {calc(y_true, y_hybrid)[0]:.4f} | Macro F1: {calc(y_true, y_hybrid)[1]:.4f} | Mango Acc: {calc(y_true, y_hybrid)[2]:.4f}")
    
    # Calculate CNN Cor->Hyb Wr and CNN Wr->Hyb Cor
    c_w = 0
    w_c = 0
    for t, c, h in zip(y_true, y_cnn, y_hybrid):
        if c == t and h != t: c_w += 1
        if c != t and h == t: w_c += 1
    
    print(f"CNN_Cor->Hyb_Wr: {c_w}")
    print(f"CNN_Wr->Hyb_Cor: {w_c}")
    
    print("\\nConfusion Matrix (CNN):")
    print(confusion_matrix(y_true, y_cnn, labels=CLASS_NAMES))
    print("\\nConfusion Matrix (Hybrid):")
    print(confusion_matrix(y_true, y_hybrid, labels=CLASS_NAMES))

if __name__ == '__main__':
    run_test()
