def calculate_hybrid_scores(cnn_probs, color_features, is_valid_mango, cnn_w=0.70, hsv_w=0.20, defect_w=0.10, defect_threshold=20.0, margin_threshold=0.10):
    """
    Hybrid Ripeness Prediction Layer
    Fuses CNN probabilities with Computer Vision HSV evidence.
    """
    # 8. Configurable fusion weights
    CNN_WEIGHT = cnn_w
    HSV_WEIGHT = hsv_w
    DEFECT_WEIGHT = defect_w
    
    # print(f"\n[HYBRID] Initializing Hybrid Ripeness Fusion...")
    # print(f"[HYBRID] Weights -> CNN: {CNN_WEIGHT}, HSV: {HSV_WEIGHT}, DEFECT: {DEFECT_WEIGHT}")
    
    # 10. Non_Mango must remain controlled by the existing object/OOD validation
    is_cnn_non_mango = cnn_probs.get('Non_Mango', 0.0) == max(cnn_probs.values())
    if not is_valid_mango or is_cnn_non_mango:
        print("[HYBRID] Object is Non_Mango. Skipping hybrid fusion.")
        return cnn_probs, "Non_Mango", 100.0, False
        
    # Extract CNN probabilities for mango classes (convert from 0-100 to 0-1)
    c_a = cnn_probs.get('Grade_A_Ripe', 0.0) / 100.0
    c_b = cnn_probs.get('Grade_B_Unripe', 0.0) / 100.0
    c_c = cnn_probs.get('Grade_C_Overripe', 0.0) / 100.0
    
    cnn_sum = c_a + c_b + c_c
    if cnn_sum > 0:
        cnn_a, cnn_b, cnn_c = c_a/cnn_sum, c_b/cnn_sum, c_c/cnn_sum
    else:
        cnn_a, cnn_b, cnn_c = 0.33, 0.33, 0.33
        
    # Extract HSV features
    yellow_pct = color_features.get('yellow_percentage', 0.0)
    green_pct = color_features.get('green_percentage', 0.0)
    dark_pct = color_features.get('dark_spots_percentage', 0.0)
    
    # 18. Scientific ripeness thresholds clearly marked as heuristic thresholds
    # --- HEURISTIC THRESHOLDS FOR HSV RIPENESS EVIDENCE ---
    total_color = yellow_pct + green_pct
    if total_color > 0:
        # Ripe evidence correlates with yellow, Unripe with green
        hsv_a = yellow_pct / total_color  
        hsv_b = green_pct / total_color   
    else:
        hsv_a, hsv_b = 0.5, 0.5
    hsv_c = 0.0 # Pure color (yellow/green) alone doesn't directly dictate overripe without defects
    
    # --- HEURISTIC THRESHOLD FOR DEFECT EVIDENCE ---
    # We assume >= threshold dark spots is strong evidence for Grade C Overripe
    defect_c = min(dark_pct / defect_threshold, 1.0)
    
    # The remaining non-defect probability supports A or B (distributed based on color)
    defect_a = (1.0 - defect_c) * hsv_a
    defect_b = (1.0 - defect_c) * hsv_b
    
    # 6. Combine CNN probability, HSV ripeness evidence, dark/decay evidence
    hybrid_a = (CNN_WEIGHT * cnn_a) + (HSV_WEIGHT * hsv_a) + (DEFECT_WEIGHT * defect_a)
    hybrid_b = (CNN_WEIGHT * cnn_b) + (HSV_WEIGHT * hsv_b) + (DEFECT_WEIGHT * defect_b)
    hybrid_c = (CNN_WEIGHT * cnn_c) + (HSV_WEIGHT * hsv_c) + (DEFECT_WEIGHT * defect_c)
    
    # Normalize final probabilities
    tot = hybrid_a + hybrid_b + hybrid_c
    if tot > 0:
        hybrid_a /= tot
        hybrid_b /= tot
        hybrid_c /= tot
    
    hybrid_probs = {
        'Grade_A_Ripe': round(hybrid_a * 100.0, 2),
        'Grade_B_Unripe': round(hybrid_b * 100.0, 2),
        'Grade_C_Overripe': round(hybrid_c * 100.0, 2),
        'Non_Mango': cnn_probs.get('Non_Mango', 0.0) # Unchanged for API compatibility
    }
    
    # Determine hybrid predicted class
    mango_classes = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe']
    hybrid_pred = max(mango_classes, key=lambda k: hybrid_probs[k])
    hybrid_conf = hybrid_probs[hybrid_pred]
    
    # 11. Confidence / Uncertainty analysis
    sorted_probs = sorted([hybrid_a, hybrid_b, hybrid_c], reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    
    # Heuristic: If the top two predictions are within margin of each other, mark as uncertain
    is_uncertain = margin < margin_threshold
    
    # 16. Detailed logging
    # print(f"[HYBRID LOG] CNN Prediction       : A={cnn_a*100:.1f}%, B={cnn_b*100:.1f}%, C={cnn_c*100:.1f}%")
    # print(f"[HYBRID LOG] HSV Evidence (Color) : A={hsv_a*100:.1f}%, B={hsv_b*100:.1f}%, C={hsv_c*100:.1f}%")
    # print(f"[HYBRID LOG] Defect Evidence      : A={defect_a*100:.1f}%, B={defect_b*100:.1f}%, C={defect_c*100:.1f}%")
    # print(f"[HYBRID LOG] Final Hybrid Prob    : A={hybrid_a*100:.1f}%, B={hybrid_b*100:.1f}%, C={hybrid_c*100:.1f}%")
    # print(f"[HYBRID LOG] Hybrid Result        : {hybrid_pred} ({hybrid_conf}%) | Uncertain: {is_uncertain}")
    
    # 12. Return original CNN probs + hybrid probs
    return hybrid_probs, hybrid_pred, hybrid_conf, is_uncertain
