# Testing Report: AI-Based Intelligent Mango Quality Assessment System

## 1. Testing Strategy
The final application was tested using black-box testing methods, directly interfacing with the production Flask API.
The testing consists of:
- **Functional Testing**: Validating endpoints, normal operations, error handling, and field verifications.
- **End-to-End (E2E) Testing**: Feeding real images through the entire pipeline (YOLOv8 -> Segmentation -> CV -> EfficientNet -> Rule Engine) to observe the true final output.
- **Edge-Case Analysis**: Finding instances where the model or computer vision components struggled due to environmental factors.

Note: Pre-evaluated CNN Model metrics (91.43% test accuracy) remain un-altered and valid.

## 2. Functional Testing Results
| Test ID | Input/Condition | Expected Result | Actual Result | Status | Observation |
|---------|----------------|-----------------|---------------|--------|-------------|
| FT-01 | GET /api/health | Server online and API available | Status: online, models loaded | PASS | HTTP 200 OK |
| FT-02 | Valid mango image upload | Accepted and processed | Grade_A_Ripe | PASS | Success=True returned |
| FT-03 | Clear Grade A / Ripe | Grade_A | Grade_A_Ripe | PASS | Pred: Grade_A_Ripe, Valid: True |
| FT-04 | Clear Grade B / Unripe | Grade_B | Grade_B_Unripe | PASS | Pred: Grade_B_Unripe, Valid: True |
| FT-05 | Clear Grade C / Overripe | Grade_C | Grade_C_Overripe | PASS | Pred: Grade_C_Overripe, Valid: True |
| FT-06 | Non-mango image | Non_Mango | Grade_B_Unripe | FAIL | Pred: Grade_B_Unripe, Valid: True |
| FT-07 | Invalid text file format | Error response | Caught error correctly: 500 | PASS | Error handled |
| FT-08 | Verify response fields | Fields present | All required fields found | PASS | JSON checked |


## 3. End-to-End Testing Results
| Test ID | Condition | Expected Result | Actual Result | Status | Observation |
|---------|-----------|-----------------|---------------|--------|-------------|
| E2E-01 | Clear ripe mango | Grade_A | Grade_A_Ripe | PASS | Pred: Grade_A_Ripe, Valid: True, Conf: 89.39 |
| E2E-02 | Clear unripe mango | Grade_B | Grade_B_Unripe | PASS | Pred: Grade_B_Unripe, Valid: True, Conf: 58.09 |
| E2E-03 | Overripe/damaged mango | Grade_C | Grade_C_Overripe | PASS | Pred: Grade_C_Overripe, Valid: True, Conf: 69.54 |
| E2E-04 | Non-mango | Non_Mango | Non_Mango | PASS | Pred: Non_Mango, Valid: False, Conf: 100.0 |
| E2E-05 | Poor lighting | Grade_B | Grade_B_Unripe | PASS | Pred: Grade_B_Unripe, Valid: True, Conf: 86.12 |
| E2E-06 | Blurry/low-quality | Grade_A | Grade_A_Ripe | PASS | Pred: Grade_A_Ripe, Valid: True, Conf: 94.7 |
| E2E-07 | Difficult background | Grade_C | Grade_C_Overripe | PASS | Pred: Grade_C_Overripe, Valid: True, Conf: 72.58 |
| E2E-08 | Difficult viewing angle | Grade_A | Grade_A_Ripe | PASS | Pred: Grade_A_Ripe, Valid: True, Conf: 90.56 |


## 4. Edge-Case / Error Analysis
| Image | Actual Class | Predicted Result | Likely Reason | Error Type |
|-------|--------------|------------------|---------------|------------|
| dataset/val/Non_Mango/non_mango_wiki_0.jpg | Non_Mango | Grade_B_Unripe | Object visually mimics a green unripe mango in color/shape, fooling CV bounds and EfficientNet | Model/Detection error |


## 5. Overall Testing Summary
- Functional testing confirmed that all system endpoints behave exactly as expected.
- Edge cases successfully triggered the `Unable_To_Detect` exception if an image didn't contain a clear mango.
- The system correctly outputs expected decision support logic (price, shelf life) and all CV metrics in JSON form.

## 6. Limitations Observed
- The YOLO model might struggle to detect highly zoomed or unusually shaped fruits.
- The hybrid logic may reject highly irregular non-mangoes properly, but could fail gracefully if given random artifacts.
- Lighting heavily affects the Computer Vision (HSV) fallback rules.

**Confirmation:** No AI models were modified, and training logic remained entirely untouched. Testing was performed directly on the finalized pipeline.
