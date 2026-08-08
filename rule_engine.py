def evaluate_mango_decision_rules(predicted_class, confidence_score, base_price_per_kg=300.0):
    """
    Rule-Based Expert System: Takes the output of the Convolutional Neural Network
    (Quality Grade & Confidence Score) and evaluates decision rules to provide:
    1. Recommended Selling Price per kg
    2. Estimated Shelf Life (Days)
    3. Actionable Storage & Retail Advice
    
    Args:
        predicted_class (str): 'Grade_A_Ripe', 'Grade_B_Unripe', or 'Grade_C_Overripe'
        confidence_score (float): Model confidence score (0.0 to 1.0)
        base_price_per_kg (float): Market benchmark price for Grade A per kg.
        
    Returns:
        dict: Rule evaluation results containing price, shelf life, discount, and advice.
    """
    confidence_pct = round(confidence_score * 100, 1)
    
    if predicted_class == 'Grade_A_Ripe':
        price_factor = 1.0  # 100% full market price
        estimated_shelf_life_days = "3 to 5 Days"
        status_category = "Premium / Optimal Freshness"
        recommendation = (
            "Display at front counters. Ideal for immediate consumption. "
            "Maintain cool, well-ventilated storage (15°C - 18°C)."
        )
        action_flag = "PRIMARY_SALE"
        
    elif predicted_class == 'Grade_B_Unripe':
        price_factor = 0.90  # 90% of base price
        estimated_shelf_life_days = "7 to 10 Days"
        status_category = "Immature / Ripening Needed"
        recommendation = (
            "Store at ambient room temperature (22°C - 25°C) to allow natural ripening. "
            "Re-grade in 3 days for potential Grade A price promotion."
        )
        action_flag = "HOLD_RIPENING"
        
    elif predicted_class == 'Grade_C_Overripe':
        price_factor = 0.50  # 50% clearance discount
        estimated_shelf_life_days = "1 Day (Immediate clearance)"
        status_category = "Overripe / Bruised / Defective"
        recommendation = (
            "Apply immediate 50% clearance discount or transfer to fruit processing (juices/puree). "
            "Isolate from fresh inventory to prevent pest or mold cross-contamination."
        )
        action_flag = "CLEARANCE_DISCOUNT"
        
    else:
        price_factor = 1.0
        estimated_shelf_life_days = "Unknown"
        status_category = "Unclassified"
        recommendation = "Manual inspection recommended."
        action_flag = "MANUAL_CHECK"

    recommended_price = round(base_price_per_kg * price_factor, 2)
    discount_pct = round((1.0 - price_factor) * 100, 0)

    return {
        "predicted_class": predicted_class,
        "confidence_percentage": confidence_pct,
        "status_category": status_category,
        "base_price": base_price_per_kg,
        "recommended_price": recommended_price,
        "discount_percentage": discount_pct,
        "estimated_shelf_life": estimated_shelf_life_days,
        "vendor_recommendation": recommendation,
        "action_flag": action_flag
    }
