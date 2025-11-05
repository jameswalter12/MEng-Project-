# Import the OpenCV library for image processing
import cv2
# Import numpy for numerical operations (not used directly here, but often useful)
import numpy as np
# Import matplotlib for displaying images
import matplotlib.pyplot as plt
import os

# Define a function to display an image using matplotlib
# 'img' is the image to display, 'cmap' sets the color map (default is grayscale)
def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))  # Create a new figure with a specific size
    ax = fig.add_subplot(111)           # Add a subplot to the figure
    if cmap:
        ax.imshow(img, cmap=cmap)       # Show the image on the subplot, 111 indicates 1x1 grid, first subplot
    else:
        ax.imshow(img)
    ax.axis('off')
    plt.show()                          # Display the figure

# The following block is commented out. It shows how to use Canny edge detection and display the result.
'''
med_value = np.median(img)  # Calculate the median pixel value of the image

lower = int(max(0,0.7*med_value))  # Set lower threshold for edge detection
upper = int(min(255,1.3*med_value))  # Set upper threshold for edge detection

blurred_img = cv2.blur(img, ksize = (6,6))  # Blur the image to reduce noise

edges = cv2.Canny(img, threshold1=lower, threshold2= upper)  # Detect edges
plt.imshow(edges)  # Show the edges
plt.axis('off')   # Hide axis
plt.show()        # Display the plot
'''

def detect_and_draw_objects(image_path, output_path=None):
    """
    Locate the red printed cube and overlay its position.
    Args:
        image_path (str): Path to the image file to process.
        output_path (str | None): Optional filepath for saving the annotated result.
    """
    # Load the current frame from disk; everything else works on this image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    # Gentle blur reduces sensor noise while keeping edges usable for contour detection
    blurred = cv2.GaussianBlur(img, (9, 9), 0)

    # HSV lets us isolate the red hue range even if brightness varies across the bed
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Red wraps the hue axis, so capture both low and high ranges; tweak if lighting changes
    lower_red1 = np.array([0, 150, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 80])
    upper_red2 = np.array([180, 255, 255])
    
    # Build a binary mask that keeps only pixels whose colour falls inside the red bands
    mask = cv2.inRange(hsv, lower_red1, upper_red1)
    mask |= cv2.inRange(hsv, lower_red2, upper_red2)

    # Close small holes and smooth jagged edges in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 7)

    # Find connected components in the mask so we can identify the cube footprint
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No red object detected. Try adjusting HSV thresholds.")
        display(mask, cmap='gray')
        return

    cube_contour = max(contours, key=cv2.contourArea) # Find the largest contour by area
    area = cv2.contourArea(cube_contour) # Calculate the area of the largest contour
    if area < 500: # Minimum area threshold to filter out noise; adjust as needed
        print(f"Detected contour is too small (area={area:.1f}); check thresholds.") #Print warning if area is too small
        display(mask, cmap='gray') # Show the mask for debugging
        return

    # Paint helpful diagnostics on a copy of the original image
    overlay = img.copy()
    x, y, w, h = cv2.boundingRect(cube_contour)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 6)

    # Analyse the interior of the cube so we can pick up the infill grid intersections
    roi = img[y:y + h, x:x + w] # Extract the region of interest (ROI) corresponding to the cube, where (x, y) is the top-left corner and (w, h) are width and height
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Convert ROI to grayscale
    roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0) # Blur the grayscale ROI to reduce noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Create CLAHE object for contrast enhancement
    roi_eq = clahe.apply(roi_gray) # Apply CLAHE to enhance contrast
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
            cv2.circle(overlay, (cx_abs, cy_abs), 6, (255, 255, 0), -1) # Draw a circle at each corner location on the overlay image
        grid_corner_count = len(corners) # Count the number of detected grid corners
    else:
        print("No grid corner features detected inside the cube. Try adjusting preprocessing.")

    M = cv2.moments(cube_contour) # Calculate spatial moments of the cube contour
    if M["m00"]:
        # Centroid from spatial moments shows the cube centre in pixel coordinates
        cx = int(M["m10"] / M["m00"]) # Calculate x coordinate of centroid
        cy = int(M["m01"] / M["m00"]) # Calculate y coordinate of centroid
        cv2.circle(overlay, (cx, cy), 20, (0, 255, 0), -1) # Draw the centroid on the overlay image
        print(f"Cube centroid at ({cx}, {cy}); bounding box {w}x{h}; area ≈ {area:.0f}") # Print cube centroid and bounding box info
    else:
        print(f"Cube bounding box {w}x{h}; area ≈ {area:.0f}")

    if grid_corner_count:
        print(f"Detected {grid_corner_count} grid corner candidates inside the cube.")

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) # Convert overlay to RGB for displaying

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) # Ensure destination directory exists
        success = cv2.imwrite(output_path, overlay)
        if success: 
            print(f"Saved annotated image to {output_path}")
        else:
            print(f"Failed to save annotated image to {output_path}")

    display(overlay_rgb, cmap=None) # Display the overlay image with detected features

# Example usage:
if __name__ == "__main__":
    # Change the paths below to suit your workspace
    detect_and_draw_objects(
        "/Users/jameswalter/Desktop/Photogammetry/Images/CUBE_1.JPG",
        output_path="/Users/jameswalter/Desktop/Photogammetry/Output/cube_with_grid.png"
    )

