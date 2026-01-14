# OpenCV drives the computer-vision pipeline used to find the target object and its surface features
import cv2
# NumPy supports pixel math and geometry calculations used by the detector
import numpy as np
# Matplotlib is used to visualize intermediate processing steps for tuning and validation
import matplotlib.pyplot as plt
import os

# This function is a preprocessing step to clean up binary masks using morphological operations which help to stabilize object detection.
def preprocess_binary_mask(mask, open_iter=1, close_iter=1, median_ksize=3):
    """Light morphological cleanup for binary masks to stabilize object detection."""
    # Sets mask to a copy to avoid modifying the original
    mask = mask.copy()
    
    # Define a 3x3 rectangular structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # If open_iter is specified, apply morphological opening to remove small noise
    if open_iter:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    
    # If close_iter is specified, apply morphological closing to fill small holes
    if close_iter:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # If median_ksize is specified and greater than 1, apply median blur to smooth edges
    if median_ksize and median_ksize > 1:
        mask = cv2.medianBlur(mask, median_ksize)
    return mask

# This function generates several binary masks based on grayscale structure cues from the input image.
def build_structure_masks(image_bgr):
    """
    Generate colour-agnostic binary masks from grayscale structure cues.
    Practical use: fallback masks when colour is unreliable, still localising the object for photogrammetry.
    Returns list of (label, mask) tuples.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    #Clahe (Contrast Limited Adaptive Histogram Equalization) enhances local contrast in the grayscale image.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Gaussian blur reduces noise and detail in the image.
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)
    
    #Otsu's thresholding automatically determines an optimal threshold value to separate foreground from background.
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive thresholding computes thresholds for smaller regions, useful for varying lighting conditions.
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    adaptive_inv = cv2.bitwise_not(adaptive)

    # Canny edge detection highlights edges in the image.
    edges = cv2.Canny(blur, 40, 120)
    edge_kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, edge_kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    
    # Return a list of labeled masks for further processing
    return [
        ("Structure mask (Otsu bright)", otsu),
        ("Structure mask (Otsu dark)", otsu_inv),
        ("Structure mask (Adaptive bright)", adaptive),
        ("Structure mask (Adaptive dark)", adaptive_inv),
        ("Structure mask (Edges)", edges),
    ]

# Display helper for step-by-step inspection during threshold/feature tuning
def display(img, cmap='gray', title=None):
    
    fig = plt.figure(figsize=(12, 10))  # Create a new figure with a specific size
    ax = fig.add_subplot(111)           # Add a subplot to the figure
    if cmap:
        ax.imshow(img, cmap=cmap)       # Show the image on the subplot, 111 indicates 1x1 grid, first subplot
    else:
        ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.show()                          # Display the figure

# Object detection and drawing function
def detect_and_draw_objects(
    image_path,
    output_path=None,
    show_steps=False,
    save_steps_dir=None,
):
    """
    Locate a region of interest from structure cues, score candidate contours, and overlay its position.
    Practical use: isolate a cube/fixture in a photogrammetry scene and extract grid corners for alignment cues.
    Args:
        image_path (str): Path to the image file to process.
        output_path (str | None): Optional filepath for saving the annotated result.
        show_steps (bool): If True, display intermediate processing steps.
        save_steps_dir (str | None): Directory for saving intermediate steps when provided.
    """
    # Load the current frame from disk; everything else works on this image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    steps = [] # List to hold intermediate processing steps for display or saving
    if show_steps: # Capture the original image for step-by-step display
        steps.append(("Original frame", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None))

    # Gentle blur reduces sensor noise while keeping edges usable for contour detection
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    if show_steps:
        steps.append(("Gaussian blur", cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), None))

    # Build binary masks from colour-agnostic structure cues
    # This keeps the detector usable across any object colour.
    mask_attempts = []
    mask_attempts.extend(build_structure_masks(blurred))

    contours = None
    mask = None
    mask_label = None
    last_processed_mask = None

    # This loops simply tries each generated mask in sequence until one yields usable contours.
    for label, raw_mask in mask_attempts:
        cleaned_mask = preprocess_binary_mask(raw_mask)
        last_processed_mask = cleaned_mask
        if show_steps:
            steps.append((f"{label} (raw)", raw_mask.copy(), 'gray'))
            steps.append((f"{label} (clean)", cleaned_mask.copy(), 'gray'))
        contours_found, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_found:
            contours = contours_found
            mask = cleaned_mask
            mask_label = label
            break
    
    # If no contours were found in any mask, exit early
    if not contours:
        print("No suitable object contour detected from HSV or structure-based masks.")
        if last_processed_mask is not None:
            if show_steps:
                for title, image, cmap in steps:
                    display(image, cmap=cmap, title=title)
            else:
                display(last_processed_mask, cmap='gray')
        return

    img_h, img_w = img.shape[:2] # Get image dimensions (height and width) to help with scoring contours
    img_cx = img_w / 2.0 # Calculate the x-coordinate of the image center 
    img_cy = img_h / 2.0 # Calculate the y-coordinate of the image center
    diag = (img_w ** 2 + img_h ** 2) ** 0.5 # Calculate the diagonal length of the image

    candidates = [] # List to hold candidate contours for the cube/fixture in the photogrammetry scene
    mask_with_boxes = None 
    for contour in contours: # Iterate over each detected contour, a countour is a list of points defining the boundary of a shape in the image
        area = cv2.contourArea(contour) # Calculate the area of the contour 
        if area < 500: # Ignore small contours that are unlikely to be the cube, 500 pixels is a reasonable threshold;
            continue

        x, y, w, h = cv2.boundingRect(contour) # Get the bounding rectangle for the contour
        bounding_area = float(w * h) # Calculate the area of the bounding rectangle 
        extent = area / bounding_area if bounding_area else 0.0 # Does the contour fill its bounding box well?

        hull = cv2.convexHull(contour) # Compute the convex hull of the contour, the smallest convex shape that encloses the contour.
        hull_area = cv2.contourArea(hull) # Calculate the area of the convex hull 
        solidity = area / hull_area if hull_area else 0.0 # Does the contour fill its convex hull well?

        aspect_ratio = w / float(h) if h else 0.0 # Calculate the aspect ratio of the bounding rectangle, which is width divided by height.
        cx = x + w / 2.0 
        cy = y + h / 2.0
        centre_dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5 # Calculate the distance from the contour center to the image center
        centre_dist_norm = centre_dist / diag if diag else 1.0 # Normalize the center distance by the image diagonal length

        candidate_mask = np.zeros_like(mask) # Create a blank mask for the candidate contour
        cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=-1) # Fill the contour on the candidate mask 
        erode_size = max(3, int(min(w, h) * 0.2)) # Erode size is the smaller of width/height * 20% or at least 3 pixels
        if erode_size % 2 == 0: # Ensure erode size is odd 
            erode_size += 1 
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size)) # Create the erosion kernel
        eroded = cv2.erode(candidate_mask, erode_kernel, iterations=1) # Erode the candidate mask to find the core area, which means shrinking the white areas in the mask
        core_area = cv2.countNonZero(eroded) # Count the non-zero pixels in the eroded mask to get the core area 
        core_ratio = core_area / area if area else 0.0 # Calculate the core ratio, which is the ratio of core area to the original contour area

        if show_steps: # Capture candidate diagnostics for step-by-step display
            candidate_id = len(candidates) + 1  # Unique ID for the candidate
            preview_scale = max(1, int(round(220.0 / max(w, h)))) if max(w, h) else 1 
            roi_preview = cv2.resize( 
                candidate_mask[y:y + h, x:x + w], 
                (max(1, w * preview_scale), max(1, h * preview_scale)),
                interpolation=cv2.INTER_NEAREST 
            )
            eroded_preview = cv2.resize(
                eroded[y:y + h, x:x + w] if eroded.ndim == 2 else eroded,
                (max(1, w * preview_scale), max(1, h * preview_scale)),
                interpolation=cv2.INTER_NEAREST
            )
            steps.append((f"Candidate {candidate_id} mask ROI", roi_preview.copy(), 'gray')) 
            steps.append((f"Candidate {candidate_id} eroded core", eroded_preview.copy(), 'gray'))

        score = extent * solidity * core_ratio # Combined score prioritizing compact, solid, well-filled shapes
        candidates.append({ 
            "contour": contour, # Store the contour itself
            "area": area, # Store the area of the contour
            "extent": extent, # Store the extent of the contour
            "solidity": solidity, # Store the solidity of the contour
            "aspect_ratio": aspect_ratio, # Store the aspect ratio of the contour
            "rect": (x, y, w, h), # Store the bounding rectangle as (x, y, width, height)
            "centre_dist": centre_dist_norm, 
            "score": score,
            "center": (cx, cy),
            "core_ratio": core_ratio,
        })

        if show_steps: # Draw bounding boxes on the mask for visualization
            if mask_with_boxes is None: 
                mask_with_boxes = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Prioritise contours that look like a filled square (inner cube) instead of thin skirts/purge lines
    # This helps lock onto the main object used for photogrammetry alignment.
    strong_square = [
        c for c in candidates # Filter candidates for strong square-like properties
        if 0.75 <= c["aspect_ratio"] <= 1.25 and c["extent"] >= 0.55 and c["solidity"] >= 0.80 # Criteria for strong square-like contours. 
    ]
    fallback = [c for c in candidates if c["extent"] >= 0.40] # Looser criteria if no strong candidates found

    ranked = strong_square or fallback # Use strong candidates if available, otherwise use fallback candidates
    if not ranked:
        print("No suitable cube contour detected; inspect the mask for tuning.") 
        if show_steps:
            if mask_with_boxes is not None:
                steps.append(("Contours (green boxes)", cv2.cvtColor(mask_with_boxes, cv2.COLOR_BGR2RGB), None))
            for title, image, cmap in steps:
                display(image, cmap=cmap, title=title)
        else:
            display(mask, cmap='gray')
        return

    ranked.sort(
        key=lambda c: (
            c["score"],
            c["extent"],
            c["solidity"],
            -c["centre_dist"],
            c["area"]
        ),
        reverse=True
    ) # Sort candidates by combined quality score, prioritising compact squares near centre
     # Select the best candidate as the detected cube
    chosen = ranked[0] # Select the top-ranked candidate as the detected cube
    cube_contour = chosen["contour"] 
    area = chosen["area"] 
    aspect = chosen["aspect_ratio"]
    extent = chosen["extent"]
    solidity = chosen["solidity"]
    centre_dist = chosen["centre_dist"]
    x, y, w, h = chosen["rect"]
    print(
        "Selected cube contour -> "
        f"area≈{area:.0f}, aspect≈{aspect:.2f}, extent≈{extent:.2f}, solidity≈{solidity:.2f}, "
        f"norm-centre-dist≈{centre_dist:.3f}"
    )

    # Paint helpful diagnostics on a copy of the original image
    overlay = img.copy() 
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255,255, 0), 2) 
    if show_steps and mask_with_boxes is not None: # Add the mask with bounding boxes to the steps for display
        steps.append(("Contours (green boxes)", cv2.cvtColor(mask_with_boxes, cv2.COLOR_BGR2RGB), None))

    # Analyse the interior of the cube so we can pick up the infill grid intersections
    # These corner features act as potential tie points for photogrammetry or quality checks.
    roi = img[y:y + h, x:x + w] # Extract the region of interest (ROI) corresponding to the cube, where (x, y) is the top-left corner and (w, h) are width and height
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Convert ROI to grayscale
    if show_steps:
        steps.append(("ROI grayscale", roi_gray.copy(), 'gray'))
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0) # Blur the grayscale ROI to reduce noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Create CLAHE object for contrast enhancement
    roi_eq = clahe.apply(roi_blur) # Apply CLAHE to enhance contrast
    if show_steps:
        steps.append(("ROI blurred", roi_blur.copy(), 'gray'))
        steps.append(("ROI CLAHE", roi_eq.copy(), 'gray'))
    corners = cv2.goodFeaturesToTrack(
        roi_eq,
        maxCorners=200,       # plenty of features to capture the lattice
        qualityLevel=0.01,    # allow moderately strong corners
        minDistance=8         # keep points separated so they trace the grid
    )
    grid_corner_count = 0 # Initialize grid corner count
    if corners is not None: # Check if any corners were detected
        for corner in corners: # Iterate over each detected corner
            cx_roi, cy_roi = corner.ravel() # Get the x, y coordinates of the corner within the ROI
            cx_abs = int(cx_roi + x) # Convert ROI coordinates to absolute image coordinates
            cy_abs = int(cy_roi + y) # Convert ROI coordinates to absolute image coordinates
            cv2.circle(overlay, (cx_abs, cy_abs), 3, (255, 255, 0), -1) # Draw a circle at each corner location on the overlay image, where 6 is the radius and (255, 255, 0) is the color (cyan). 
        grid_corner_count = len(corners) # Count the number of detected grid corners
    else:
        print("No grid corner features detected inside the cube. Try adjusting preprocessing.")

    M = cv2.moments(cube_contour) # Calculate spatial moments of the cube contour
    if M["m00"]:
        # Centroid from spatial moments shows the cube centre in pixel coordinates
        cx = int(M["m10"] / M["m00"]) # Calculate x coordinate of centroid
        cy = int(M["m01"] / M["m00"]) # Calculate y coordinate of centroid
        cv2.circle(overlay, (cx, cy), 3, (0, 255, 0), -1) # Draw the centroid on the overlay image
        print(f"Cube centroid at ({cx}, {cy}); bounding box {w}x{h}; area ≈ {area:.0f}") # Print cube centroid and bounding box info
    else:
        print(f"Cube bounding box {w}x{h}; area ≈ {area:.0f}") # Print bounding box info if centroid cannot be calculated

    # This finalizes the annotated overlay image and handles output
    if grid_corner_count: 
        print(f"Detected {grid_corner_count} grid corner candidates inside the cube.") 

    # This converts the overlay to RGB for displaying
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) # Convert overlay to RGB for displaying
    if show_steps:
        steps.append(("Annotated result", overlay_rgb.copy(), None))

    # Save the annotated overlay image if an output path is provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) # Ensure destination directory exists
        success = cv2.imwrite(output_path, overlay)
        if success:
            print(f"Saved annotated image to {output_path}")
        else:
            print(f"Failed to save annotated image to {output_path}")

    # Prepare the result dictionary with relevant outputs
    result = {
        "overlay_bgr": overlay,
        "overlay_rgb": overlay_rgb,
        "roi_bgr": roi,
        "roi_gray": roi_gray,
        "roi_processed": roi_eq,
        "roi_rect": (x, y, w, h),
        "cube_contour": cube_contour,
        "mask": mask,
    }
    # Display intermediate steps or the final overlay based on user preference
    if show_steps:
        for title, image, cmap in steps:
            display(image, cmap=cmap, title=title)
        if save_steps_dir:
            print("Step export disabled; skipping save_steps_dir output for now.")
    else:
        display(overlay_rgb, cmap=None) # Display the overlay image with detected features

    return result

# Main execution block for testing the object detection pipeline
if __name__ == "__main__":
    cube_image_path = "/Users/jameswalter/Desktop/Photogammetry/Images/TEST.jpg"
    grid_image_path = "/Users/jameswalter/Desktop/Photogammetry/Images/GRID_CHECK.png"
    overlay_output_path = "/Users/jameswalter/Desktop/Photogammetry/Output/cube_with_grid_2.png"
    steps_output_dir = "/Users/jameswalter/Desktop/Photogammetry/Output/steps"

    # First pass: detect the cube/fixture and mark grid corners in a reference photo.
    # This creates a stable ROI for later feature matching between frames.
    first_result = detect_and_draw_objects(
        cube_image_path,
        output_path=overlay_output_path,
        show_steps=True,
        save_steps_dir=steps_output_dir,
    )
    if first_result is None:
        raise SystemExit("Detector failed on the first image; adjust preprocessing or file path.")

    # Second pass: detect another frame/angle; same pipeline to extract a comparable ROI.
    second_result = detect_and_draw_objects(
        grid_image_path,
        output_path=None,
        show_steps=False,
        save_steps_dir=None,
    )
    if second_result is None:
        raise SystemExit("Detector failed on the second image; adjust preprocessing or thresholds.")

    roi_1_processed = first_result["roi_processed"]
    roi_2_processed = second_result["roi_processed"]

    # ORB features provide scale/rotation-invariant keypoints for matching between views.
    # In photogrammetry this is a starting point for alignment or quality comparison.
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(roi_1_processed, None)
    kp2, des2 = orb.detectAndCompute(roi_2_processed, None)
    # S
    print(f"ROI1 shape {roi_1_processed.shape}, keypoints: {0 if kp1 is None else len(kp1)}")
    print(f"ROI2 shape {roi_2_processed.shape}, keypoints: {0 if kp2 is None else len(kp2)}")

    # Match ORB descriptors between the two ROIs
    if des1 is None or des2 is None:
        print("Not enough ORB keypoints detected in one of the ROIs.")
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches.sort(key=lambda m: m.distance)

        keep = matches[:30]
        print(f"Total ORB matches found: {len(matches)}; displaying top {len(keep)}")

        roi_1_display = cv2.cvtColor(first_result["roi_bgr"], cv2.COLOR_BGR2RGB)
        roi_2_display = cv2.cvtColor(second_result["roi_bgr"], cv2.COLOR_BGR2RGB)
        
        # If no matches were found, display the ROIs instead
        if not keep:
            print("No matches to display; showing the two ROIs instead.")
            display(roi_1_display, cmap=None, title="First ROI (no matches)")
            display(roi_2_display, cmap=None, title="Second ROI (no matches)")
        else:
            match_vis = cv2.drawMatches(
                roi_1_display,
                kp1,
                roi_2_display,
                kp2,
                keep,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            match_vis_rgb = cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)
            display(match_vis_rgb, cmap=None, title="Top ORB feature matches between ROIs")
