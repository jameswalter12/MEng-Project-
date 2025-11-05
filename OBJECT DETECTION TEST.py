# Import the OpenCV library for image processing
import cv2
# Import numpy for numerical operations (not used directly here, but often useful)
import numpy as np
# Import matplotlib for displaying images
import matplotlib.pyplot as plt

# Define a function to display an image using matplotlib
# 'img' is the image to display, 'cmap' sets the color map (default is grayscale)
def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))  # Create a new figure with a specific size
    ax = fig.add_subplot(111)           # Add a subplot to the figure
    ax.imshow(img, cmap=cmap)           # Show the image on the subplot
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

def detect_and_draw_objects(image_path):
    """
    Loads an image, processes it to detect objects/contours, draws them, and displays the result.
    Args:
        image_path (str): Path to the image file to process.
    """
    # Read an image from disk
    img  = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    # Apply a median blur to the image to reduce noise
    phone_blur = cv2.medianBlur(img, 25)
    # Convert the blurred image to grayscale (single channel)
    grey_phone = cv2.cvtColor(phone_blur, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to the grayscale image
    # Pixels above the threshold become 0 (black), below become 255 (white), using Otsu's method
    ret, thresh = cv2.threshold(grey_phone, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Create a small matrix (kernel) of ones for morphological operations
    kernel = np.ones((3,3), np.uint8)
    # Use morphological opening to remove small white noise from the thresholded image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Dilate the image to increase the white region (background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Find contours (outlines) in the thresholded image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through all found contours
    if hierarchy is not None:
        for i in range(len(contours)):
            # Only draw the outermost contours (those with no parent)
            if hierarchy[0][i][3] == -1:
                # Draw the contour on the original image in blue with thickness 10
                cv2.drawContours(img, contours, i, (255, 0, 0), 10)
    # Display the final image with contours drawn
    display(img)

# Example usage:
if __name__ == "__main__":
    # Change the path below to any image you want to process
    detect_and_draw_objects("/Users/jameswalter/Desktop/Photogammetry/Images/CUBE_1.JPG")


