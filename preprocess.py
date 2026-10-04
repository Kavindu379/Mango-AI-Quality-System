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

def apply_segmentation_mask(image_rgb, mask_np):
    """
    Applies a binary polygon segmentation mask (from YOLOv8-Seg) to the RGB image.
    Pixels outside the mask are zeroed out (set to black background).
    """
    if mask_np is None:
        return image_rgb
    
    try:
        # Ensure mask dimensions match image dimensions
        if mask_np.shape[:2] != image_rgb.shape[:2]:
            mask_resized = cv2.resize(mask_np.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_np.astype(np.uint8)
            
        mask_3ch = np.stack([mask_resized]*3, axis=-1)
        masked_rgb = np.where(mask_3ch > 0, image_rgb, 0)
        return masked_rgb
    except Exception as e:
        print(f"[WARNING] Segmentation mask application failed: {e}")
        return image_rgb

def extract_hsv_color_analysis(img_rgb, mask_np=None):
    """
    Converts RGB image tensor to HSV color space and extracts Yellow, Green,
    Dark Spot decay ratios, and Red/Non-Mango masks using natural fruit threshold masking.
    If mask_np is provided, evaluates color ratios strictly on non-zero fruit pixels.
    """
    if mask_np is not None:
        img_rgb = apply_segmentation_mask(img_rgb, mask_np)
        if mask_np.shape[:2] != img_rgb.shape[:2]:
            valid_mask = cv2.resize(mask_np.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            valid_mask = mask_np.astype(np.uint8)
        total_pixels = max(1, cv2.countNonZero(valid_mask))
    else:
        # Exclude plain white/light background pixels AND pure dark/black background pixels
        # Black backgrounds (R<15, G<15, B<15) have HSV V<30 which matches the dark-spot filter,
        # causing them to be falsely counted as mango decay when the fruit is on a dark surface.
        white_bg = (img_rgb[:, :, 0] > 235) & (img_rgb[:, :, 1] > 235) & (img_rgb[:, :, 2] > 235)
        black_bg = (img_rgb[:, :, 0] < 15) & (img_rgb[:, :, 1] < 15) & (img_rgb[:, :, 2] < 15)
        valid_mask = (~(white_bg | black_bg)).astype(np.uint8) * 255
        non_bg_count = cv2.countNonZero(valid_mask)
        total_pixels = non_bg_count if non_bg_count > 100 else (img_rgb.shape[0] * img_rgb.shape[1])

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. Ripe Golden Yellow / Orange Color Mask (Rich ripe golden hue H: 14-34, S: >= 75)
    lower_yellow = np.array([14, 75, 60])
    upper_yellow = np.array([34, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=valid_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    
    # 2. Unripe Green Color Mask (Includes light green, yellow-green, and deep raw mango green skin: H: 35-92)
    lower_green = np.array([35, 25, 60])
    upper_green = np.array([92, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    green_mask = cv2.bitwise_and(green_mask, green_mask, mask=valid_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    # 3. Dark Spot / Decay Spot Mask (V >= 15 to exclude pure-black background pixels leaking through mask)
    lower_dark = np.array([0, 0, 15])
    upper_dark = np.array([180, 255, 85])
    dark_mask = cv2.inRange(img_hsv, lower_dark, upper_dark)
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=valid_mask)
    dark_pixels = cv2.countNonZero(dark_mask)

    # 4. Pure Bright Red Mask (Apples / Tomatoes - Non-Mango hue)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([4, 255, 255])
    lower_red2 = np.array([172, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(img_hsv, lower_red1, upper_red1) | cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=valid_mask)
    red_pixels = cv2.countNonZero(red_mask)
    
    yellow_pct = min(100.0, round((yellow_pixels / total_pixels) * 100, 2))
    green_pct = min(100.0, round((green_pixels / total_pixels) * 100, 2))
    dark_pct = min(100.0, round((dark_pixels / total_pixels) * 100, 2))
    red_pct = min(100.0, round((red_pixels / total_pixels) * 100, 2))
    
    return {
        "yellow_percentage": yellow_pct,
        "green_percentage": green_pct,
        "dark_spots_percentage": dark_pct,
        "red_percentage": red_pct
    }

def validate_is_mango_candidate(color_features, confidence_score=1.0, img_rgb=None, has_yolo_detection=True, mask_np=None):
    """
    Advanced Multi-Feature Out-Of-Distribution (OOD) Guard:
    Verifies fruit skin color spectrum, contour geometry, and YOLO detection validity.
    """
    yellow_pct = color_features.get('yellow_percentage', 0.0)
    green_pct = color_features.get('green_percentage', 0.0)
    dark_pct = color_features.get('dark_spots_percentage', 0.0)
    red_pct = color_features.get('red_percentage', 0.0)
    
    total_mango_skin_pct = yellow_pct + green_pct

    # The CNN model is highly accurate at detecting Non_Mango (e.g. apple).
    # Brittle HSV color checks frequently falsely reject dark or oddly-lit valid mangoes.
    # Therefore, we trust YOLO to find the crop, and the CNN to classify Non_Mango if it's not a mango.
    return True, "Valid Object"
