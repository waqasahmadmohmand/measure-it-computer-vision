# Project Plan — Measure It

## 1. Project Overview

**Project:** Measure It — Segmentation and Metric Measurement  
**Objective:** Develop a computer vision system that detects a book in an image and estimates its physical width and height in millimetres.

The project will use a mobile phone camera, OpenCV camera calibration, a custom-labelled image dataset, and a lightweight U-Net segmentation model.

## 2. Hardware and Environment

- Processor: CPU (no dedicated GPU)
- RAM: 8 GB
- Camera: Mobile phone
- Target object: Physical book
- Language: Python
- Deep learning framework: PyTorch
- Image processing: OpenCV

## 3. Implementation Approach

The work will follow this sequence:

1. **Camera Calibration:** Capture at least 20 checkerboard images and calculate the camera matrix, distortion coefficients and reprojection error using OpenCV.

2. **Dataset Collection:** Capture at least 70 original photographs of the book using fixed camera settings and varied backgrounds, positions and lighting conditions.

3. **Dataset Annotation:** Create binary segmentation masks using CVAT. Establish fixed training, validation and test splits of 70%, 20% and 10%, respectively.

4. **Image Preprocessing:** Undistort the captured images using the saved camera calibration parameters and prepare the images and masks for training.

5. **Model Development:** Implement and train a lightweight U-Net segmentation model using PyTorch. The architecture is selected to reduce computational requirements on an 8 GB CPU-only system.

6. **Model Evaluation:** Evaluate the held-out test dataset using mask IoU, precision, recall, mAP@0.5 and mAP@0.5:0.95.

7. **Measurement Pipeline:** Build a single-command application that undistorts a new image, segments the book, detects a known-size ArUco reference marker and calculates physical dimensions.

8. **Accuracy Validation:** Compare automated measurements with physical measurements obtained using a ruler or calliper across at least 10 instances.

9. **Experimental Testing:** Compare measurements from raw and undistorted images and investigate the effect of one selected condition, such as camera tilt.

10. **Reliability and Documentation:** Establish measurement rejection thresholds, document limitations, and prepare reproducible instructions and a demonstration recording.

## 4. Time Allocation

The assessment will be completed within the eight-hour limit.

- 0–30 minutes: Repository and documentation.
- 30–90 minutes: Camera and dataset preparation.
- 90–180 minutes: Calibration and preprocessing.
- 180–300 minutes: Annotation and model training.
- 300–360 minutes: Measurement pipeline.
- 360–420 minutes: Accuracy and experimental testing.
- 420–480 minutes: Documentation, verification and recording.

Actual completion times and deviations will be recorded.

## 5. Constraints and Risks

- CPU-only training may limit model complexity and training speed.
- Eight GB RAM requires controlled image resolution and batch size.
- Inaccurate segmentation can affect measurement results.
- The reference marker and book must occupy the same measurement plane.
- Camera tilt and perspective may introduce measurement errors.
- Unfinished objectives will be documented rather than represented as completed.

## 6. Deferred Work

If time or hardware constraints prevent completion, additional model optimization, extensive environmental testing and advanced perspective correction will be deferred and documented.

## 7. Engineering Principle

Measurement accuracy and reproducibility take priority over model complexity. All reported results must originate from actual experiments and be supported by saved outputs.