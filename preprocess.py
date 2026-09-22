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
        # Exclude plain white/light background canvas pixels (R>235, G>235, B>235) from total count
        valid_mask = (~((img_rgb[:, :, 0] > 235) & (img_rgb[:, :, 1] > 235) & (img_rgb[:, :, 2] > 235))).astype(np.uint8) * 255
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
    
    # 3. Dark Spot / Decay Spot Mask (Increased Value threshold to catch brown rot)
    lower_dark = np.array([0, 0, 0])
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

    # Rejection Rule 1: Pure Red Non-Mango Object (Apples / Tomatoes)
    if not has_yolo_detection and red_pct > 35.0 and yellow_pct < 10.0 and green_pct < 10.0:
        return False, "Non-Mango Object Detected: Skin color signature matches Apple / Red non-mango fruit."

    # Rejection Rule 2: Insufficient Mango skin color detected
    if total_mango_skin_pct < 2.0:
        return False, "Insufficient Mango skin color detected (Less than 2%). Please scan a valid Mango."

    # Rejection Rule 3: No YOLO detection box and low fruit skin metrics
    if not has_yolo_detection and total_mango_skin_pct < 10.0 and dark_pct < 5.0:
        return False, "No Mango object detected in image frame. Please center a valid Mango in camera view."

    # Rejection Rule 4: Rotated & Straight Aspect Ratio Geometry check (catches pencils, pens, sticks at any angle)
    if img_rgb is not None and img_rgb.shape[0] > 50 and img_rgb.shape[1] > 50:
        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            if mask_np is not None:
                # Use perfect YOLO mask contour
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # Find foreground object contour (non-white/dark background)
                _, thresh = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # FORENSIC CHECK 1: MORPHOLOGICAL OUTLINE & GEOMETRY (Shape Verification)
                
                # A. Oriented Bounding Box Aspect Ratio (Banana check)
                rect = cv2.minAreaRect(largest_contour)
                (cx, cy), (w_rect, h_rect), angle = rect
                min_dim = min(w_rect, h_rect)
                max_dim = max(w_rect, h_rect)
                if min_dim > 0:
                    true_aspect_ratio = float(max_dim) / min_dim
                    # Mangoes are round/oval. Bananas are elongated curved arcs.
                    if true_aspect_ratio > 2.2:
                        return False, "Failed Check 1: Banana curved arc geometry and longitudinal ridge lines detected"

                # B. Circularity (Citrus / Apple check)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.88: # Highly spherical circle
                        return False, "Failed Check 1: Citrus/Apple spherical circle or indented heart-shaped top/bottom stem detected"
                
                # C. Straight Bounding Box Extreme Aspect Ratio (Pencils, sticks)
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > 0 and h > 0:
                    aspect_ratio = float(w) / h
                    if aspect_ratio > 3.2 or aspect_ratio < 0.31:
                        return False, f"Invalid Object Geometry (Bounding Box Aspect Ratio: {aspect_ratio:.2f}). Please scan a Mango."

                # D. Convexity / Solidity Check (Banana Bunch / Open Hand / Irregular objects)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and area > 0:
                    solidity = area / float(hull_area)
                    # Mangoes are extremely solid convex ovals (> 95%). Banana bunches have large gaps (< 85%).
                    if solidity < 0.85:
                        return False, f"Failed Check 1: Irregular non-convex shape (Solidity: {solidity:.2f}). Possible banana bunch or clustered object."

                # FORENSIC CHECK 2: SURFACE TEXTURE & STRUCTURAL RIDGES (Skin Inspection)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                variance_laplacian = np.var(laplacian[mask == 255])
                
                # Reject if Banana uniform flat lemon-yellow (high yellow %, very smooth)
                if yellow_pct > 80.0 and variance_laplacian < 50:
                    return False, "Failed Check 2: Banana uniform flat lemon-yellow color without natural mango blush gradients detected"
                
                # Reject if Citrus bumpy/dimpled texture (very high laplacian variance)
                if variance_laplacian > 1500 and total_mango_skin_pct > 0:
                    return False, "Failed Check 2: Citrus bumpy, dimpled, or porous oil-gland skin texture detected"

        except Exception as ge:
            print(f"[WARNING] Geometry validation error: {ge}")
            pass

    return True, "Valid Mango Object Detected"
