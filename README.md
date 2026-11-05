
# Measure It — Segmentation and Metric Measurement

A computer vision project that aims to segment a physical object from an image and estimate its dimensions in millimetres using a calibrated camera and a known-size reference object.

## Project Status

**Current stage:** Camera calibration and undistortion are complete; manual mask annotation is next. Model training, measurement inference and calliper validation are not complete.

### Implemented
- Project objectives and completion criteria documented.
- Initial project plan documented.
- Fixed 70/20/10 dataset split lists created and committed.
- Checkerboard camera calibration implemented using OpenCV.
- Calibration dataset collected: 24 images.
- Successful checkerboard detections: 21 out of 24.
- Camera matrix and distortion coefficients saved.
- 70 original book images collected at 2296 × 4080 pixels.
- All 70 images undistorted into `dataset/images_undistorted/`; source images are unchanged.

### Hardware and Data

- Camera: Samsung Galaxy A06 (reported by the author).
- Compute: CPU-only, 8 GB RAM, Windows (processor model and GPU details not yet recorded).
- Object: book; physical dimensions and selection rationale still need to be recorded.
- Reference object: ArUco marker; its physical dimensions still need to be measured and recorded.
- Split: 49 train / 14 validation / 7 test, fixed in `dataset/splits/`.
- Annotation tool and masks: not yet completed.

## Dataset Preprocessing

Run from the repository root after capturing all dataset images at the calibrated
resolution:

```powershell
.\.venv\Scripts\python.exe dataset\undistort_dataset.py
```

The script loads `calibration/calibration_data.npz`, checks that all 70 source
images are readable and match the calibrated resolution, then writes undistorted
copies with the original filenames to `dataset/images_undistorted/`. It fails
before writing if any source image has an incompatible resolution. Label these
undistorted images so each mask aligns with the image used for training.

## Manual Mask Annotation

Run the local OpenCV polygon annotator from the repository root:

```powershell
.\.venv\Scripts\python.exe dataset\annotate_book.py
```

For each image, left-click around the **visible rigid book boundary**, excluding
the reference card, loose wrapping and shadows. Do not invent a hidden edge; skip
an ambiguous image and revisit or retake it. Use the mouse wheel to zoom around
the cursor, right-click to center the view, `U` to undo, `R` to reset, `N` to save
and move to the next image, `S` to skip, and `Q` or `Esc` to quit. Masks are saved
as binary PNG files in `dataset/masks/` at the source resolution. The script
checks the committed split filenames and resumes by skipping existing masks.
Review every saved mask before training; skipped images are not considered
annotated.

### Initial Calibration Results

| Metric | Result |
|---|---:|
| Checkerboard inner corners | 9 × 7 |
| Square size | 20 mm |
| Calibration images | 24 |
| Successful detections | 21 |
| Detection rate | 87.5% |
| Image resolution | 2296 × 4080 |
| OpenCV RMS reprojection error | 2.2359 pixels |
| Mean reprojection error | 1.7436 pixels |
| Maximum reprojection error | 3.2611 pixels |

These are initial calibration results. Measurement accuracy has not yet been validated.

## Technology Stack

- Python
- OpenCV
- NumPy
- PyTorch (planned for segmentation model training)

## Repository Structure

```text
measure-it-computer-vision/
├── calibration/
│   ├── images/
│   ├── calibrate_camera.py
│   └── calibration_data.npz
├── dataset/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── docs/
│   ├── Plan.md
│   ├── Objectives.md
│   └── Definition_of_Done.md
├── README.md
└── requirements.txt