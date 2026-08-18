def evaluate_mango_decision_rules(predicted_class, confidence_score, base_price_per_kg=300.0):
    """
    Rule-Based Expert Inference Engine for Mango Quality & Pricing.
    Now handles 'Non_Mango' invalid object detection cleanly.
    """
    if predicted_class == 'Non_Mango':
        return {
            "status_category": "Invalid Object",
            "is_valid_mango": False,
            "recommended_price": 0.0,
            "discount_percentage": 100,
            "estimated_shelf_life": "N/A",
            "vendor_recommendation": "⚠️ Non-Mango or Unknown Object Detected! Please scan a valid Ripe, Unripe, or Overripe Mango.",
            "storage_guidance": "Do not process non-fruit items."
        }
        
    if predicted_class == 'Grade_A_Ripe':
        price_multiplier = 1.0
        discount_pct = 0
        shelf_life = "3 to 5 Days"
        status_cat = "Premium Fresh"
        recommendation = "Optimal quality for immediate sale. Place on front counter display. Store at 15°C–18°C."
        storage = "Cool ambient environment (15°C–18°C). Avoid direct sunlight."
        
    elif predicted_class == 'Grade_B_Unripe':
        price_multiplier = 0.90
        discount_pct = 10
        shelf_life = "7 to 10 Days"
        status_cat = "Immature / Ripening Stock"
        recommendation = "Hold stock for 3 to 4 days to allow natural ripening. Store at room temp (22°C–25°C)."
        storage = "Room temperature storage (22°C–25°C). Keep well ventilated."
        
    elif predicted_class == 'Grade_C_Overripe':
        price_multiplier = 0.50
        discount_pct = 50
        shelf_life = "1 Day (Immediate Sale Needed)"
        status_cat = "Clearance / Discount Required"
        recommendation = "Apply 50% clearance discount for quick sale today or transfer to juice processing."
        storage = "Isolate immediately from fresh stock to prevent mold transfer. Refrigerate at 10°C."
        
    else:
        price_multiplier = 1.0
        discount_pct = 0
        shelf_life = "Unknown"
        status_cat = "Uncertain"
        recommendation = "Re-inspect fruit under better lighting."
        storage = "Standard ambient storage."

    recommended_price = round(base_price_per_kg * price_multiplier, 2)

    return {
        "status_category": status_cat,
        "is_valid_mango": True,
        "recommended_price": recommended_price,
        "discount_percentage": discount_pct,
        "estimated_shelf_life": shelf_life,
        "vendor_recommendation": recommendation,
        "storage_guidance": storage
    }
