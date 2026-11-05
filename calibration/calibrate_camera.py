
import cv2
import numpy as np
from pathlib import Path

# Checkerboard configuration
PATTERN_SIZE = (9, 7)  # Inner corners: columns, rows
SQUARE_SIZE_MM = 20.0

IMAGE_DIR = Path("calibration/images")
OUTPUT_FILE = Path("calibration/calibration_data.npz")

# Prepare 3D coordinates of checkerboard corners
objp = np.zeros(
    (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:PATTERN_SIZE[0],
    0:PATTERN_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE_MM

object_points = []
image_points = []

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

if len(image_files) < 20:
    raise RuntimeError(
        f"Need at least 20 calibration images. Found {len(image_files)}."
    )

image_size = None

for image_path in image_files:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    current_size = (gray.shape[1], gray.shape[0])

    if image_size is None:
        image_size = current_size
    elif current_size != image_size:
        raise ValueError("All images must have the same resolution.")

    # Detect checkerboard corners
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, PATTERN_SIZE
        )
    else:
        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

    if found:
        object_points.append(objp.copy())
        image_points.append(corners)

        print(f"Detected: {image_path.name}")

    else:
        print(f"Not detected: {image_path.name}")

print("\nSuccessful detections:", len(object_points))

if len(object_points) < 20:
    raise RuntimeError(
        "At least 20 successful checkerboard detections are required."
    )

# Calculate camera calibration
rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)

# Calculate mean reprojection error
errors = []

for i in range(len(object_points)):

    projected, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = np.linalg.norm(
        image_points[i].reshape(-1, 2) -
        projected.reshape(-1, 2),
        axis=1
    ).mean()

    errors.append(error)

mean_error = float(np.mean(errors))
max_error = float(np.max(errors))

# Save calibration parameters
np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_size=image_size,
    rms=rms,
    mean_reprojection_error=mean_error,
    max_reprojection_error=max_error,
    square_size_mm=SQUARE_SIZE_MM,
    pattern_size=PATTERN_SIZE
)

print("\n========== CALIBRATION RESULTS ==========")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
print(f"OpenCV RMS Error: {rms:.4f} pixels")
print(f"Mean Reprojection Error: {mean_error:.4f} pixels")
print(f"Maximum Reprojection Error: {max_error:.4f} pixels")
print("Image Resolution:", image_size)
print("Calibration Saved:", OUTPUT_FILE)
