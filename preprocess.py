import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

TARGET_SIZE = (224, 224)

# PyTorch Image Normalization transform
PYTORCH_TRANSFORM = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image_pytorch(image_input):
    """
    Standardizes image and converts it into a PyTorch Tensor shape (1, 3, 224, 224).
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('RGB')
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert('RGB')
    else:
        raise TypeError("Unsupported image input type.")
        
    tensor_img = PYTORCH_TRANSFORM(pil_img)
    tensor_batch = tensor_img.unsqueeze(0)  # Add batch dimension: (1, 3, 224, 224)
    
    # Also return numpy RGB array (224, 224, 3) for OpenCV color feature extraction and display
    resized_pil = pil_img.resize(TARGET_SIZE)
    img_rgb_np = np.array(resized_pil)
    
    return tensor_batch, img_rgb_np

def extract_hsv_color_analysis(image_rgb):
    """
    Computer Vision Feature Extraction: Analyzes HSV color channels to estimate 
    the ratio of Green (Unripe), Yellow/Orange (Ripe), and Dark/Brown spots (Overripe).
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Green range (Unripe)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Yellow/Orange range (Ripe)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Dark/Brown spots (Overripe/Spoiled)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    
    green_pct = (np.count_nonzero(mask_green) / total_pixels) * 100
    yellow_pct = (np.count_nonzero(mask_yellow) / total_pixels) * 100
    dark_pct = (np.count_nonzero(mask_dark) / total_pixels) * 100
    
    return {
        "green_percentage": round(green_pct, 2),
        "yellow_percentage": round(yellow_pct, 2),
        "dark_spots_percentage": round(dark_pct, 2)
    }
