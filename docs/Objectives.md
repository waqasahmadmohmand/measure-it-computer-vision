# Project Objectives — Measure It

## MO-1: Physical Measurement Accuracy

**What is measured:** Difference between automated book dimensions and physical ground-truth measurements.

**Method:** Compare automated width and height against ruler or calliper measurements across at least 10 measurement instances.

**Target:** Mean absolute percentage error (MAPE) of 5% or less.

**Conditions:** Fixed camera settings, book and reference marker on the same plane, controlled capture conditions.

**Not included:** Partially visible books, unreliable reference detection and severe perspective distortion.

## MO-2: Segmentation Performance

**What is measured:** Quality of the predicted book segmentation masks.

**Method:** Evaluate predicted masks against manually annotated ground-truth masks on the fixed test dataset.

**Target:** Mean test mask IoU of at least 0.85.

**Conditions:** Seven or more held-out test images, with no test-image use during training or model selection.

**Not included:** Images outside the defined dataset conditions.

## Evaluation and Reporting

The final report will include actual measured results, the per-instance accuracy table, model evaluation metrics, hardware specifications and any objectives not achieved.

Targets are engineering goals, not claimed results. All deviations will be documented.