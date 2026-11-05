# Definition of Done — Measure It

## Camera Calibration
- [ ] At least 20 calibration images captured.
- [ ] Camera matrix and distortion coefficients saved.
- [ ] Reprojection error calculated and reported.
- [ ] Undistortion visually verified.

## Dataset
- [ ] At least 70 original images collected.
- [ ] Every image has a corresponding segmentation mask.
- [ ] Train/validation/test split fixed and committed.
- [ ] No test images used for training or model selection.

## Segmentation Model
- [ ] Lightweight U-Net implemented.
- [ ] Model trained on the training split.
- [ ] Test IoU, precision and recall calculated.
- [ ] mAP@0.5 and mAP@0.5:0.95 reported.

## Measurement Pipeline
- [ ] Single-command inference implemented.
- [ ] Image undistortion integrated.
- [ ] Book segmentation implemented.
- [ ] Reference marker detection implemented.
- [ ] Width and height reported in millimetres.
- [ ] Object-not-found and reference-not-found cases handled.

## Validation
- [ ] At least 10 physical measurement comparisons completed.
- [ ] MAE and MAPE reported.
- [ ] Raw versus undistorted comparison completed.
- [ ] One failure condition experimentally investigated.
- [ ] Measurement rejection threshold evaluated.

## Submission
- [ ] README completed.
- [ ] Plan, Objectives and Definition of Done committed.
- [ ] Model configuration and reproducibility information included.
- [ ] Model weights and SHA-256 hash provided.
- [ ] Git history contains meaningful incremental commits.
- [ ] Pull request created from dev-test01 into main.
- [ ] Loom recording is no longer than five minutes.

**Evidence:** Every completed item must reference an actual result, file, commit or recording. Unfinished items must be explicitly identified.