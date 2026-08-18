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
        color_features = extract_hsv_color_analysis(img_rgb)
        
        # 1. HARD PRE-FILTER: Verify skin color threshold
        is_valid_mango, validation_msg = validate_is_mango_candidate(color_features, 1.0, img_rgb)
        
        pred_class = 'Grade_A_Ripe'
        conf = 0.95
        class_probs = {}
        has_yolo_detection = False

        if not is_valid_mango:
            pred_class = 'Non_Mango'
            conf = 0.99
            rule_results = evaluate_mango_decision_rules('Non_Mango', 0.99, base_price_per_kg=base_price)
            return jsonify({
                "success": True,
                "is_valid_mango": False,
                "prediction": {
                    "class_code": "Non_Mango",
                    "display_name": CLASS_DISPLAY_NAMES.get("Non_Mango", "Invalid Object / Not a Mango 🚫"),
                    "confidence_percentage": 99.0,
                    "class_probabilities": {"Non_Mango": 99.0}
                },
                "computer_vision_features": color_features,
                "rule_engine": rule_results
            })

        # 2. Run YOLOv8 Inference with Standard Confidence Threshold (conf=0.35)
        if y_m is not None:
            results = y_m(img_rgb, conf=0.35, verbose=False)
            if len(results[0].boxes) > 0:
                top_box = results[0].boxes[0]
                conf = float(top_box.conf[0].cpu().item())
                cls_id = int(top_box.cls[0].cpu().item())
                
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
                
        # 3. Fallback to MobileNetV2 PyTorch model if YOLO is not active
        elif p_m is not None:
            with torch.no_grad():
                logits = p_m(tensor_batch)
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
        else:
            y_pct = color_features['yellow_percentage']
            g_pct = color_features['green_percentage']
            d_pct = color_features['dark_spots_percentage']
            if d_pct > 3.0:
                pred_class = 'Grade_C_Overripe'
                conf = 0.88
            elif g_pct > y_pct:
                pred_class = 'Grade_B_Unripe'
                conf = 0.91
            else:
                pred_class = 'Grade_A_Ripe'
                conf = 0.95
            class_probs = {pred_class: conf * 100}
            has_yolo_detection = True

        rule_results = evaluate_mango_decision_rules(pred_class, conf, base_price_per_kg=base_price)
        
        return jsonify({
            "success": True,
            "is_valid_mango": is_valid_mango,
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
