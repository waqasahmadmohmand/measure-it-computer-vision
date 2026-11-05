from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "images_undistorted"
MASK_DIR = ROOT / "dataset" / "masks"
SPLIT_DIR = ROOT / "dataset" / "splits"
SPLIT_FILES = ("train.txt", "val.txt", "test.txt")
WINDOW_NAME = "Book mask annotation"
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 900
IMAGE_TOP = 42
IMAGE_BOTTOM = 830
MIN_ZOOM = 1.0
MAX_ZOOM = 12.0


def create_binary_mask(
    image_shape: tuple[int, ...], vertices: list[tuple[int, int]]
) -> np.ndarray:
    if len(vertices) < 3:
        raise ValueError("A polygon needs at least three vertices.")

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def load_split_names(image_names: set[str]) -> set[str]:
    split_lists = [
        (SPLIT_DIR / filename).read_text(encoding="utf-8").splitlines()
        for filename in SPLIT_FILES
    ]
    combined = [name for names in split_lists for name in names if name]

    if len(combined) != len(set(combined)):
        raise ValueError("Dataset split lists contain duplicate image names.")
    if set(combined) != image_names:
        raise ValueError(
            "The committed split filenames must match the undistorted images."
        )
    return set(combined)


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Undistorted image directory not found: {IMAGE_DIR}")

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No JPEG images found in {IMAGE_DIR}.")

    load_split_names({path.name for path in image_paths})
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    pending = []
    for image_path in image_paths:
        mask_path = MASK_DIR / f"{image_path.stem}.png"
        if mask_path.exists():
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"Existing mask does not match its image: {mask_path.name}"
                )
            continue
        pending.append((image_path, mask_path))

    print(f"Images to label: {len(pending)} of {len(image_paths)}")
    print("Trace only the visible rigid book boundary; exclude the card and shadows.")
    print("Left click: add vertex | wheel: zoom | right click: center view")
    print("U: undo | R: reset polygon | N: save and next | S: skip | Q/Esc: quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)

    for index, (image_path, mask_path) in enumerate(pending, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]
        state: dict[str, object] = {
            "image": image,
            "height": height,
            "width": width,
            "vertices": [],
            "zoom": MIN_ZOOM,
            "center_x": width / 2,
            "center_y": height / 2,
            "view": None,
        }

        def image_point(screen_x: int, screen_y: int) -> tuple[int, int] | None:
            view = state["view"]
            if view is None:
                return None
            x0, y0, scale, left, top, shown_width, shown_height = view
            if not (x0 <= screen_x < x0 + shown_width):
                return None
            if not (y0 <= screen_y < y0 + shown_height):
                return None
            x = round(left + (screen_x - x0) / scale)
            y = round(top + (screen_y - y0) / scale)
            return (
                max(0, min(width - 1, x)),
                max(0, min(height - 1, y)),
            )

        def mouse_callback(
            event: int, x: int, y: int, flags: int, _parameter: object
        ) -> None:
            vertices = state["vertices"]
            if event == cv2.EVENT_LBUTTONDOWN:
                point = image_point(x, y)
                if point is not None:
                    vertices.append(point)
            elif event == cv2.EVENT_RBUTTONDOWN:
                point = image_point(x, y)
                if point is not None:
                    state["center_x"], state["center_y"] = point
            elif event == cv2.EVENT_MOUSEWHEEL:
                old_point = image_point(x, y)
                old_zoom = float(state["zoom"])
                new_zoom = min(
                    MAX_ZOOM,
                    max(MIN_ZOOM, old_zoom * (1.25 if flags > 0 else 0.8)),
                )
                if old_point is not None and new_zoom != old_zoom:
                    view = state["view"]
                    if view is not None:
                        x0, y0, _, _, _, shown_width, shown_height = view
                        rel_x = (x - x0) / shown_width
                        rel_y = (y - y0) / shown_height
                        new_view_width = width / new_zoom
                        new_view_height = height / new_zoom
                        state["center_x"] = old_point[0] + (
                            0.5 - rel_x
                        ) * new_view_width
                        state["center_y"] = old_point[1] + (
                            0.5 - rel_y
                        ) * new_view_height
                state["zoom"] = new_zoom

        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        while True:
            zoom = float(state["zoom"])
            view_width = max(1, min(width, round(width / zoom)))
            view_height = max(1, min(height, round(height / zoom)))
            center_x = float(state["center_x"])
            center_y = float(state["center_y"])
            left = max(0, min(width - view_width, round(center_x - view_width / 2)))
            top = max(0, min(height - view_height, round(center_y - view_height / 2)))
            crop = image[top : top + view_height, left : left + view_width]

            available_width = CANVAS_WIDTH
            available_height = IMAGE_BOTTOM - IMAGE_TOP
            scale = min(
                available_width / view_width,
                available_height / view_height,
            )
            shown_width = max(1, round(view_width * scale))
            shown_height = max(1, round(view_height * scale))
            offset_x = (CANVAS_WIDTH - shown_width) // 2
            offset_y = IMAGE_TOP + (available_height - shown_height) // 2

            canvas = np.full(
                (CANVAS_HEIGHT, CANVAS_WIDTH, 3), 35, dtype=np.uint8
            )
            resized = cv2.resize(crop, (shown_width, shown_height))
            canvas[
                offset_y : offset_y + shown_height,
                offset_x : offset_x + shown_width,
            ] = resized
            state["view"] = (
                offset_x,
                offset_y,
                scale,
                left,
                top,
                shown_width,
                shown_height,
            )

            vertices = state["vertices"]
            display_vertices = [
                (
                    round(offset_x + (px - left) * scale),
                    round(offset_y + (py - top) * scale),
                )
                for px, py in vertices
            ]
            for point in display_vertices:
                cv2.circle(canvas, point, 4, (0, 0, 255), -1)
            if len(display_vertices) > 1:
                cv2.polylines(
                    canvas,
                    [np.asarray(display_vertices, dtype=np.int32)],
                    False,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                canvas,
                f"{index}/{len(pending)}  {image_path.name}  zoom={zoom:.1f}x",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Visible rigid book only | N save  U undo  R reset  S skip  Q quit",
                (12, 860),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(20) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("Stopped. Existing masks were kept; rerun to resume.")
                return
            if key == ord("u") and vertices:
                vertices.pop()
            elif key == ord("r"):
                vertices.clear()
            elif key == ord("s"):
                print(f"Skipped {image_path.name}")
                break
            elif key == ord("n"):
                if len(vertices) < 3:
                    print("Add at least three polygon vertices before saving.")
                    continue
                mask = create_binary_mask(image.shape, vertices)
                if not cv2.imwrite(str(mask_path), mask):
                    raise OSError(f"Could not save mask: {mask_path}")
                print(f"Saved {mask_path.relative_to(ROOT)}")
                break

    cv2.destroyAllWindows()
    print("Annotation pass finished. Skipped images still need labels.")


if __name__ == "__main__":
    main()
