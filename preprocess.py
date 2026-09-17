import cv2
import numpy as np
import torch
from torchvision import transforms

def preprocess_image_pytorch(image_pil):
    """
    Standardizes image to 224x224 RGB tensor for PyTorch MobileNetV2 inference.
    """
    img_rgb = np.array(image_pil)
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform_pipeline(image_pil).unsqueeze(0) # Shape: (1, 3, 224, 224)
    return tensor, img_rgb

def extract_hsv_color_analysis(img_rgb):
    """
    Converts RGB image tensor to HSV color space and extracts Yellow, Green,
    Dark Spot decay ratios, and Red/Non-Mango masks using natural fruit threshold masking.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
    
    # 1. Ripe Yellow / Orange Color Mask (Vibrant Fruit Saturation H: 16-38, S: >= 110 - Excludes Human Skin Tones)
    lower_yellow = np.array([16, 110, 60])
    upper_yellow = np.array([38, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    
    # 2. Unripe Green Color Mask (Includes light & deep mango green skin)
    lower_green = np.array([34, 30, 30])
    upper_green = np.array([88, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(green_mask)
    
    # 3. Dark Spot / Decay Spot Mask
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(img_hsv, lower_dark, upper_dark)
    dark_pixels = cv2.countNonZero(dark_mask)

    # 4. Pure Bright Red Mask (Apples / Tomatoes - Non-Mango hue)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([4, 255, 255])
    lower_red2 = np.array([172, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(img_hsv, lower_red1, upper_red1) | cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_pixels = cv2.countNonZero(red_mask)
    
    yellow_pct = round((yellow_pixels / total_pixels) * 100, 2)
    green_pct = round((green_pixels / total_pixels) * 100, 2)
    dark_pct = round((dark_pixels / total_pixels) * 100, 2)
    red_pct = round((red_pixels / total_pixels) * 100, 2)
    
    return {
        "yellow_percentage": yellow_pct,
        "green_percentage": green_pct,
        "dark_spots_percentage": dark_pct,
        "red_percentage": red_pct
    }

def validate_is_mango_candidate(color_features, confidence_score=1.0, img_rgb=None, has_yolo_detection=True):
    """
    Advanced Multi-Feature Out-Of-Distribution (OOD) Guard:
    Verifies fruit skin color spectrum, contour geometry, and YOLO detection validity.
    """
    yellow_pct = color_features.get('yellow_percentage', 0.0)
    green_pct = color_features.get('green_percentage', 0.0)
    dark_pct = color_features.get('dark_spots_percentage', 0.0)
    red_pct = color_features.get('red_percentage', 0.0)
    
    total_mango_skin_pct = yellow_pct + green_pct

    # Rejection Rule 1: Pure Red Non-Mango Object (Apples / Tomatoes)
    if red_pct > 35.0 and yellow_pct < 10.0 and green_pct < 10.0:
        return False, "Non-Mango Object Detected: Skin color signature matches Apple / Red non-mango fruit."

    # Rejection Rule 2: Insufficient Mango skin color detected
    if total_mango_skin_pct < 2.0 and dark_pct < 1.0:
        return False, "Insufficient Mango skin color detected (Less than 2%). Please scan a valid Mango."

    # Rejection Rule 3: No YOLO detection box and low fruit skin metrics
    if not has_yolo_detection and total_mango_skin_pct < 10.0 and dark_pct < 5.0:
        return False, "No Mango object detected in image frame. Please center a valid Mango in camera view."

    # Rejection Rule 4: Aspect Ratio & Contour Geometry check
    if img_rgb is not None and img_rgb.shape[0] > 50 and img_rgb.shape[1] > 50:
        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > 0 and h > 0:
                    aspect_ratio = float(w) / h
                    # Extreme line or ultra-flat rectangle shapes (pens, books)
                    if aspect_ratio > 3.5 or aspect_ratio < 0.25:
                        return False, f"Invalid Object Geometry (Aspect Ratio: {aspect_ratio:.2f}). Please scan a Mango."
        except Exception:
            pass

    return True, "Valid Mango Object Detected"
