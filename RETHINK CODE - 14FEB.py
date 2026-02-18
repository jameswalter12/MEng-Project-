import argparse
import os
import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_GREEN_HSV_RANGE = "35,20,20:105,255,255"

def preprocess_binary_mask(mask, open_iter=1, close_iter=1, median_ksize=3):
    """Light cleanup for binary masks."""
    cleaned = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if open_iter:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if median_ksize and median_ksize > 1:
        cleaned = cv2.medianBlur(cleaned, median_ksize)
    return cleaned


def fill_holes_with_outer_contour(mask):
    """Fill holes by drawing only external contours as solid regions."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    solid = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(solid, contours, -1, 255, thickness=cv2.FILLED)
    return preprocess_binary_mask(solid, open_iter=0, close_iter=2, median_ksize=3)


def build_structure_edge_mask(image_bgr):
    """Color-agnostic edge mask used to support HSV masking."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return preprocess_binary_mask(edges, open_iter=0, close_iter=2, median_ksize=3)

def build_hsv_mask(image_bgr, hsv_ranges):
    """Build a binary mask from one or more HSV ranges."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    combined = None
    for lower, upper in hsv_ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        current = cv2.inRange(hsv, lower_np, upper_np)
        combined = current if combined is None else cv2.bitwise_or(combined, current)
    return preprocess_binary_mask(combined, open_iter=1, close_iter=2, median_ksize=3)

def pick_hsv_roi(
    hsv_mask,
    target_center_frac=(0.5, 0.5),
    min_area_frac=0.01,
    max_area_frac=0.75,
    max_aspect_ratio=3.5,
):
    """Pick an ROI from HSV mask using area/aspect/center scoring."""
    h, w = hsv_mask.shape[:2]
    total_area = float(h * w)
    diag = (h * h + w * w) ** 0.5
    target_x = w * target_center_frac[0]
    target_y = h * target_center_frac[1]

    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        area_frac = area / total_area
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 3 or bh < 3:
            continue
        aspect = max(bw, bh) / float(min(bw, bh))
        if aspect > max_aspect_ratio:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < 0.3:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        center_dist = (((cx - target_x) ** 2 + (cy - target_y) ** 2) ** 0.5) / diag
        center_weight = max(0.0, 1.0 - center_dist)

        score = (0.45 * area_frac) + (0.25 * solidity) + (0.30 * center_weight)
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh, contour)

    if best is None:
        return None, None
    x, y, bw, bh, contour = best
    return (x, y, bw, bh), contour


def show_result_popup(original_bgr, hsv_mask, edges_mask, selected_mask, masked_image_bgr):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    panels = [
        ("Original", cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), None),
        ("HSV mask", hsv_mask, "gray"),
        ("Structure edge mask", edges_mask, "gray"),
        ("Selected object mask", selected_mask, "gray"),
        ("Masked full frame", cv2.cvtColor(masked_image_bgr, cv2.COLOR_BGR2RGB), None),
    ]
    for idx, (title, frame, cmap) in enumerate(panels):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        ax.set_title(title)
        if cmap:
            ax.imshow(frame, cmap=cmap)
        else:
            ax.imshow(frame)
        ax.axis("off")
    axes[1, 2].axis("off")
    fig.tight_layout()
    plt.show()

def mask_single_image(
    image_path,
    output_path,
    hsv_ranges,
    show_popup=False,
    target_center_frac=(0.5, 0.5),
    min_area_frac=0.005,
    max_area_frac=0.85,
    max_aspect_ratio=6.0,
):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    hsv_mask = build_hsv_mask(blurred, hsv_ranges)
    roi_rect, contour = pick_hsv_roi(
        hsv_mask,
        target_center_frac=target_center_frac,
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
        max_aspect_ratio=max_aspect_ratio,
    )
    if roi_rect is None or contour is None:
        raise RuntimeError("No HSV ROI passed filters. Adjust HSV ranges or geometry thresholds.")

    selected_mask = np.zeros_like(hsv_mask, dtype=np.uint8)
    cv2.drawContours(selected_mask, [contour], -1, 255, thickness=cv2.FILLED)
    selected_mask = preprocess_binary_mask(
        selected_mask, open_iter=0, close_iter=3, median_ksize=5
    )
    selected_mask = fill_holes_with_outer_contour(selected_mask)

    # Keep original resolution to stabilize SfM/camera handling.
    masked_image = cv2.bitwise_and(image, image, mask=selected_mask)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not cv2.imwrite(output_path, masked_image):
        raise RuntimeError(f"Failed to write masked image to: {output_path}")

    if show_popup:
        # Debug-only panel, not used for final mask creation.
        edges_mask = build_structure_edge_mask(blurred)
        show_result_popup(image, hsv_mask, edges_mask, selected_mask, masked_image)


def list_images_in_folder(folder_path, recursive=False):
    image_paths = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for name in sorted(files):
                full = os.path.join(root, name)
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(full)
    else:
        for name in sorted(os.listdir(folder_path)):
            full = os.path.join(folder_path, name)
            if not os.path.isfile(full):
                continue
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(full)
    return image_paths


def run_mask_batch(
    input_path,
    output_dir,
    hsv_ranges,
    show_popup=False,
    recursive=False,
    target_center_frac=(0.5, 0.5),
    min_area_frac=0.005,
    max_area_frac=0.85,
    max_aspect_ratio=6.0,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = list_images_in_folder(input_path, recursive=recursive)
        if not image_paths:
            raise RuntimeError(f"No images found in folder: {input_path}")

    success_count = 0
    failures = []

    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_masked_roi.png")
        try:
            mask_single_image(
                image_path=image_path,
                output_path=output_path,
                hsv_ranges=hsv_ranges,
                show_popup=show_popup,
                target_center_frac=target_center_frac,
                min_area_frac=min_area_frac,
                max_area_frac=max_area_frac,
                max_aspect_ratio=max_aspect_ratio,
            )
            success_count += 1
            print(f"[OK]   {os.path.basename(image_path)} -> {output_path}")
        except Exception as exc:
            failures.append((image_path, str(exc)))
            print(f"[FAIL] {os.path.basename(image_path)} -> {exc}")

    print(f"\nBatch complete: {success_count} succeeded, {len(failures)} failed.")
    return failures


def parse_hsv_ranges(hsv_specs):
    """
    Parse repeated --hsv-range args like:
    --hsv-range 35,20,20:105,255,255
    """
    parsed = []
    for spec in hsv_specs:
        try:
            lower_text, upper_text = spec.split(":")
            lower = tuple(int(v) for v in lower_text.split(","))
            upper = tuple(int(v) for v in upper_text.split(","))
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError
            parsed.append((lower, upper))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --hsv-range '{spec}'. Expected format: H,S,V:H,S,V"
            ) from exc
    return parsed

def main():
    parser = argparse.ArgumentParser(
        description="Mask-only batch pipeline (full-frame masked outputs; no SIFT / no point cloud)."
    )
    parser.add_argument("--input-path", type=str, required=True, help="Image file or folder to process.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save *_masked_roi.png outputs.")
    parser.add_argument(
        "--hsv-range",
        action="append",
        default=[DEFAULT_GREEN_HSV_RANGE],
        help="HSV range in H,S,V:H,S,V form. Repeat flag for multiple ranges.",
    )
    parser.add_argument("--show-popup", action="store_true", help="Show per-image preview popup.")
    parser.add_argument("--recursive", action="store_true", help="Process images in subfolders too.")
    parser.add_argument("--target-center-x", type=float, default=0.5, help="Target ROI center x fraction.")
    parser.add_argument("--target-center-y", type=float, default=0.5, help="Target ROI center y fraction.")
    parser.add_argument("--min-area-frac", type=float, default=0.005, help="Minimum ROI area fraction.")
    parser.add_argument("--max-area-frac", type=float, default=0.85, help="Maximum ROI area fraction.")
    parser.add_argument("--max-aspect-ratio", type=float, default=6.0, help="Maximum ROI aspect ratio.")
    args = parser.parse_args()

    hsv_ranges = parse_hsv_ranges(args.hsv_range)
    run_mask_batch(
        input_path=args.input_path,
        output_dir=args.output_dir,
        hsv_ranges=hsv_ranges,
        show_popup=args.show_popup,
        recursive=args.recursive,
        target_center_frac=(args.target_center_x, args.target_center_y),
        min_area_frac=args.min_area_frac,
        max_area_frac=args.max_area_frac,
        max_aspect_ratio=args.max_aspect_ratio,
    )


if __name__ == "__main__":
    main()

'''
RUN EXAMPLE: 

python3 'RETHINK CODE - 14FEB.py' \
  --input-path '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/RAW_IMAGES' \
  --output-dir '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/IMAGES_PROCESSED' \
  --recursive

 WITH SPECIFIC HSV RANGES: 

python3 'RETHINK CODE - 14FEB.py' \
  --input-path '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/RAW_IMAGES' \
  --output-dir '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/IMAGES_PROCESSED' \
  --hsv-range 35,20,20:105,255,255

 EXAMPLE 3) 

 python3 'RETHINK CODE - 14FEB.py' \
  --input-path '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/RAW_IMAGES' \
  --output-dir '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/IMAGES_PROCESSED' \
  --recursive \
  --hsv-range 35,20,20:105,255,255 \
  --min-area-frac 0.003 \
  --max-area-frac 0.90 \
  --max-aspect-ratio 7.5
 
EXAMPLE 4) 

python3 'RETHINK CODE - 14FEB.py' \
  --input-path '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/RAW_IMAGES' \
  --output-dir '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/IMAGES_PROCESSED' \
  --recursive \
  --hsv-range 35,20,20:105,255,255 \
  --min-area-frac 0.003 \
  --max-area-frac 0.90 \
  --max-aspect-ratio 7.5

'''
