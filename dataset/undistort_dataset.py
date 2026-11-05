from pathlib import Path

import cv2
import numpy as np


SOURCE_DIR = Path("dataset/images")
OUTPUT_DIR = Path("dataset/images_undistorted")
CALIBRATION_FILE = Path("calibration/calibration_data.npz")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_FILE}")
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Dataset image directory not found: {SOURCE_DIR}")

    with np.load(CALIBRATION_FILE) as calibration:
        camera_matrix = calibration["camera_matrix"]
        distortion_coefficients = calibration["dist_coeffs"]
        expected_size = tuple(int(value) for value in calibration["image_size"])

    image_paths = sorted(
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_paths) != 70:
        raise RuntimeError(f"Expected 70 dataset images; found {len(image_paths)}.")

    validated_images = []
    invalid_images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            invalid_images.append(f"{image_path.name}: unreadable")
            continue

        actual_size = (image.shape[1], image.shape[0])
        if actual_size != expected_size:
            invalid_images.append(
                f"{image_path.name}: resolution {actual_size}, "
                f"expected {expected_size}"
            )
            continue
        validated_images.append((image_path, image))

    if invalid_images:
        details = "\n".join(f"- {item}" for item in invalid_images)
        raise ValueError(
            "Dataset images must be readable and match the calibrated "
            f"resolution before any output is written:\n{details}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path, image in validated_images:
        undistorted = cv2.undistort(
            image, camera_matrix, distortion_coefficients
        )
        output_path = OUTPUT_DIR / image_path.name
        if not cv2.imwrite(str(output_path), undistorted):
            raise OSError(f"Could not write undistorted image: {output_path}")

    print(
        f"Undistorted {len(validated_images)} images "
        f"at resolution {expected_size}."
    )
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
