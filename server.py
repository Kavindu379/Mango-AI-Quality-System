import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from preprocess import preprocess_image_pytorch, extract_hsv_color_analysis, validate_is_mango_candidate
from model import build_mango_cnn_model, CLASS_NAMES, CLASS_DISPLAY_NAMES
from rule_engine import evaluate_mango_decision_rules

app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
CORS(app)

PYTORCH_MODEL_PATH = 'mango_model.pth'
YOLO_MODEL_PATH = 'best.pt'
TEST_IMAGES_DIR = 'test_images'

pytorch_model = None
yolo_model = None

def get_or_load_models():
    global pytorch_model, yolo_model
    
    if yolo_model is None and os.path.exists(YOLO_MODEL_PATH):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("[SUCCESS] Loaded YOLOv8 Object Detection & Quality model (best.pt)!")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8 model: {e}")

    if pytorch_model is None and os.path.exists(PYTORCH_MODEL_PATH):
        try:
            m = build_mango_cnn_model(num_classes=len(CLASS_NAMES))
            m.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')), strict=False)
            m.eval()
            pytorch_model = m
            print("[SUCCESS] Loaded PyTorch Neural Network model weights (mango_model.pth).")
        except Exception as e:
            print(f"[ERROR] Failed to load PyTorch model weights: {e}")

    return yolo_model, pytorch_model

@app.route('/')
def index():
    return send_from_directory('frontend/dist', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    y_m, p_m = get_or_load_models()
    return jsonify({
        "status": "online",
        "yolo_loaded": y_m is not None,
        "pytorch_loaded": p_m is not None,
        "classes": CLASS_NAMES
    })

@app.route('/api/samples', methods=['GET'])
def list_sample_images():
    samples = []
    if os.path.exists(TEST_IMAGES_DIR):
        for fname in sorted(os.listdir(TEST_IMAGES_DIR)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                label = fname.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title()
                samples.append({
                    "filename": fname,
                    "label": label,
                    "url": f"/api/samples/{fname}"
                })
    return jsonify({"samples": samples})

@app.route('/api/predict', methods=['POST'])
def predict_mango():
    y_m, p_m = get_or_load_models()
    base_price = request.form.get('base_price', default=300.0, type=float)
    image = None
    
    if 'image' in request.files:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
    elif 'sample_name' in request.form:
        sample_name = request.form.get('sample_name')
        sample_path = os.path.join(TEST_IMAGES_DIR, sample_name)
        if os.path.exists(sample_path):
            image = Image.open(sample_path).convert('RGB')
        else:
            return jsonify({"error": f"Sample image '{sample_name}' not found."}), 404
    elif request.is_json and 'base64_image' in request.json:
        b64_data = request.json['base64_image'].split(',')[-1]
        img_bytes = base64.b64decode(b64_data)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    else:
        return jsonify({"error": "No image file, sample_name, or base64_image provided."}), 400

    try:
        tensor_batch, img_rgb = preprocess_image_pytorch(image)
        
        cropped_mango_rgb = img_rgb
        pred_class = 'Grade_A_Ripe'
        conf = 0.95
        class_probs = {}
        has_yolo_detection = False
        bounding_box_coords = None

        # 1. Run YOLOv8 Inference & Auto-Crop Bounding Box Region
        if y_m is not None:
            results = y_m(img_rgb, conf=0.35, verbose=False)
            if len(results[0].boxes) > 0:
                top_box = results[0].boxes[0]
                conf = float(top_box.conf[0].cpu().item())
                cls_id = int(top_box.cls[0].cpu().item())
                
                # Extract Bounding Box Coordinates [x1, y1, x2, y2]
                xyxy = top_box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(img_rgb.shape[1], xyxy[2]), min(img_rgb.shape[0], xyxy[3])
                
                # AUTO-CROP MANGO REGION: Removes background hands, tables, & room clutter!
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    cropped_mango_rgb = img_rgb[y1:y2, x1:x2]
                    bounding_box_coords = [int(x1), int(y1), int(x2), int(y2)]
                    print(f"[INFO] YOLOv8 Auto-Cropped Mango Region: Box [{x1}, {y1}, {x2}, {y2}]")

                # Roboflow dataset class mapping: 0 -> Grade_A_Ripe, 1 -> Grade_B_Unripe, 2 -> Grade_C_Overripe, 3 -> Grade_A_Ripe
                if cls_id == 1:
                    pred_class = 'Grade_B_Unripe'
                elif cls_id == 2:
                    pred_class = 'Grade_C_Overripe'
                else:
                    pred_class = 'Grade_A_Ripe'
                
                class_probs = {pred_class: round(conf * 100, 2)}
                has_yolo_detection = True
            else:
                pred_class = 'Non_Mango'
                conf = 0.99
                has_yolo_detection = False

        # 2. Extract OpenCV HSV features on the CROPPED MANGO region (100% clean fruit pixels!)
        color_features = extract_hsv_color_analysis(cropped_mango_rgb)

        # 3. Hybrid OpenCV HSV + AI Classification Fusion Rule:
        # Physical feature analysis overrides CNN if severe rot/dark spots or green immaturity is detected!
        dark_spots_pct = color_features.get('dark_spots_percentage', 0.0)
        green_pct = color_features.get('green_percentage', 0.0)
        yellow_pct = color_features.get('yellow_percentage', 0.0)

        if dark_spots_pct >= 10.0:
            pred_class = 'Grade_C_Overripe'
            conf = max(conf, 0.92)
            class_probs = {'Grade_C_Overripe': round(conf * 100, 2), 'Grade_A_Ripe': 5.0, 'Grade_B_Unripe': 1.0, 'Non_Mango': 0.0}
            print(f"[HYBRID AI FUSION] Severe Dark Spots detected ({dark_spots_pct}%) -> Classified as Grade_C_Overripe!")
        elif green_pct >= 45.0 and yellow_pct < 15.0 and dark_spots_pct < 8.0:
            pred_class = 'Grade_B_Unripe'
            conf = max(conf, 0.90)
            class_probs = {'Grade_B_Unripe': round(conf * 100, 2), 'Grade_A_Ripe': 8.0, 'Grade_C_Overripe': 2.0, 'Non_Mango': 0.0}
            print(f"[HYBRID AI FUSION] Predominantly Green skin ({green_pct}%) -> Classified as Grade_B_Unripe!")

        # 4. Hybrid Out-Of-Distribution (OOD) Guard
        is_valid_mango, validation_msg = validate_is_mango_candidate(color_features, conf, cropped_mango_rgb)
        
        # 5. Fallback to MobileNetV2 PyTorch model if YOLO is not active
        if y_m is None and p_m is not None:
            # Preprocess the cropped mango for MobileNetV2
            cropped_pil = Image.fromarray(cropped_mango_rgb)
            cropped_tensor, _ = preprocess_image_pytorch(cropped_pil)
            
            with torch.no_grad():
                logits = p_m(cropped_tensor)
                probabilities = F.softmax(logits, dim=1).numpy()[0]
                pred_index = int(np.argmax(probabilities))
                if pred_index < len(CLASS_NAMES):
                    pred_class = CLASS_NAMES[pred_index]
                    conf = float(probabilities[pred_index])
                class_probs = {
                    CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
                    for i in range(min(len(CLASS_NAMES), len(probabilities)))
                }
            has_yolo_detection = True

        # 4. Enforce Hybrid CV + Deep Learning Feature Fusion Rule
        # If OpenCV physical feature extraction detects over 10.0% dark spot decay/rot coverage,
        # override prediction to Grade_C_Overripe regardless of YOLO bounding box label!
        dark_spots_pct = color_features.get('dark_spots_percentage', 0.0)
        green_pct = color_features.get('green_percentage', 0.0)
        yellow_pct = color_features.get('yellow_percentage', 0.0)

        if is_valid_mango and pred_class != 'Non_Mango':
            if dark_spots_pct >= 10.0:
                pred_class = 'Grade_C_Overripe'
                conf = max(conf, 0.95)
                class_probs = {'Grade_C_Overripe': round(conf * 100, 2), 'Grade_A_Ripe': 5.0, 'Grade_B_Unripe': 1.0}
                print(f"[HYBRID AI FUSION] Severe Dark Spots detected ({dark_spots_pct}%) -> Overriding to Grade_C_Overripe!")
            elif green_pct >= 45.0 and yellow_pct < 15.0 and dark_spots_pct < 8.0:
                pred_class = 'Grade_B_Unripe'
                conf = max(conf, 0.90)
                class_probs = {'Grade_B_Unripe': round(conf * 100, 2), 'Grade_A_Ripe': 8.0, 'Grade_C_Overripe': 2.0}
                print(f"[HYBRID AI FUSION] Predominantly Green skin ({green_pct}%) -> Overriding to Grade_B_Unripe!")

        # Enforce Non_Mango if skin color features fail OR 0 YOLO boxes detected
        if not is_valid_mango or (y_m is not None and not has_yolo_detection) or pred_class == 'Non_Mango':
            pred_class = 'Non_Mango'
            conf = 0.99
            is_valid_mango = False

        rule_results = evaluate_mango_decision_rules(pred_class, conf, base_price_per_kg=base_price)
        
        return jsonify({
            "success": True,
            "is_valid_mango": is_valid_mango,
            "bounding_box": bounding_box_coords,
            "prediction": {
                "class_code": pred_class,
                "display_name": CLASS_DISPLAY_NAMES.get(pred_class, pred_class),
                "confidence_percentage": round(conf * 100, 2),
                "class_probabilities": class_probs
            },
            "computer_vision_features": color_features,
            "rule_engine": rule_results
        })
        
    except Exception as e:
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500

@app.route('/api/samples/<filename>', methods=['GET'])
def get_sample_image(filename):
    return send_from_directory(TEST_IMAGES_DIR, filename)

if __name__ == '__main__':
    get_or_load_models()
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask AI Backend Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
