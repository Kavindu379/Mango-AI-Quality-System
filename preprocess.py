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
    and Dark Spot decay ratios using natural fruit threshold masking.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
    
    # 1. Ripe Yellow / Orange Color Mask (Includes natural orange & red blush tones)
    lower_yellow = np.array([5, 30, 30])
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
    
    yellow_pct = round((yellow_pixels / total_pixels) * 100, 2)
    green_pct = round((green_pixels / total_pixels) * 100, 2)
    dark_pct = round((dark_pixels / total_pixels) * 100, 2)
    
    return {
        "yellow_percentage": yellow_pct,
        "green_percentage": green_pct,
        "dark_spots_percentage": dark_pct
    }

def validate_is_mango_candidate(color_features, confidence_score=1.0, img_rgb=None):
    """
    Balanced Out-Of-Distribution (OOD) Guard:
    Ensures real mangoes pass while filtering plain empty backgrounds.
    """
    yellow_pct = color_features['yellow_percentage']
    green_pct = color_features['green_percentage']
    dark_pct = color_features['dark_spots_percentage']
    
    total_mango_skin_pct = yellow_pct + green_pct
    
    # Rejection if less than 2% yellow/green skin color is present
    if total_mango_skin_pct < 2.0 and dark_pct < 1.0:
        return False, "Insufficient Mango skin color detected (Less than 2%). Please scan a valid Mango."
        
    return True, "Valid Mango Object Detected"
