import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from preprocess import preprocess_image_pytorch, extract_hsv_color_analysis
from model import build_mango_cnn_model, CLASS_NAMES, CLASS_DISPLAY_NAMES
from rule_engine import evaluate_mango_decision_rules

app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
CORS(app)  # Enable Cross-Origin Resource Sharing for React Frontend

MODEL_PATH = 'mango_model.pth'
TEST_IMAGES_DIR = 'test_images'
model = None

def get_or_load_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            m = build_mango_cnn_model()
            m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            m.eval()
            model = m
            print("[SUCCESS] Loaded PyTorch Neural Network model weights.")
        except Exception as e:
            print(f"[ERROR] Failed to load model weights: {e}")
    return model

@app.route('/api/health', methods=['GET'])
def health_check():
    m = get_or_load_model()
    return jsonify({
        "status": "online",
        "model_loaded": m is not None,
        "classes": CLASS_NAMES
    })

@app.route('/api/samples', methods=['GET'])
def list_sample_images():
    """Dynamically lists all image files inside the test_images folder."""
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
    m = get_or_load_model()
    
    # Extract Base Price from form data or JSON query
    base_price = request.form.get('base_price', default=300.0, type=float)
    
    image = None
    
    # 1. Check if image file was uploaded
    if 'image' in request.files:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
    # 2. Check if sample_name was passed
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
        # Preprocess Image
        tensor_batch, img_rgb = preprocess_image_pytorch(image)
        
        # Computer Vision HSV Feature Extraction
        color_features = extract_hsv_color_analysis(img_rgb)
        
        # Neural Network Inference
        if m is not None:
            with torch.no_grad():
                logits = m(tensor_batch)
                probabilities = F.softmax(logits, dim=1).numpy()[0]
                pred_index = int(np.argmax(probabilities))
                pred_class = CLASS_NAMES[pred_index]
                conf = float(probabilities[pred_index])
                
                class_probs = {
                    CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
                    for i in range(len(CLASS_NAMES))
                }
        else:
            # Fallback based on HSV color analysis if model weights file missing
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

        # Rule Engine Decision Evaluation
        rule_results = evaluate_mango_decision_rules(pred_class, conf, base_price_per_kg=base_price)
        
        return jsonify({
            "success": True,
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
    get_or_load_model()
    print("[INFO] Starting Flask AI Backend Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
