# Import the OpenCV library for image processing
import cv2
# Import numpy for numerical operations (not used directly here, but often useful)
import numpy as np
# Import matplotlib for displaying images
import matplotlib.pyplot as plt
'''
photo = cv2.imread("/Users/jameswalter/Desktop/Photogammetry/Images/CUBE_1.JPG")
grey_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) 


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))  # Create a new figure with a specific size
    ax = fig.add_subplot(111)           # Add a subplot to the figure
    if cmap:
        ax.imshow(img, cmap=cmap)       # Show the image on the subplot
    else:
        ax.imshow(img)
    plt.show()                          # Display the figure 


# Starting with Harris corner detection

grey = np.float32(grey_photo)  # Convert image to float32 type for Harris detector
dst = cv2.cornerHarris(src = grey, blockSize = 2, ksize = 3, k = 0.04)  # Apply Harris corner detection

#Here blocksize is neighbourhood size (corner eigenvalues and vecros), ksize is aperture parameter (lernel size) for Sobel operator, k is Harris detector free parameter

dst = cv2.dilate(dst, None)  # Dilate the result to mark the corners

corner_mask = dst > 0.1 * dst.max()  # Threshold to get the strongest corner responses
print(f"Detected {corner_mask.sum()} Harris corner pixels")

# Draw detected corners in red on top of the original colour image
overlay = photo.copy()
overlay[corner_mask] = [0, 0, 255]

# Matplotlib expects RGB ordering
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

display(overlay_rgb, cmap=None)  # Display the image with corners marked in red


# Now using Shi-Tomasi corner detection
photo = cv2.imread("/Users/jameswalter/Desktop/Photogammetry/Images/CUBE_1.JPG")
photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(photo_gray, 64, 0.01, 10) # Detect up to 64 corners with a quality level of 0.01 and minimum distance of 10 pixels. 

corners = np.int8(corners)  # Convert corner coordinates to integer type

for i in corners:
    x, y = i.ravel()  # Flatten the array and get x, y coordinates
    cv2.circle(photo_gray, (x, y), 3,
                (0, 0, 255), -1)  # Draw a blue circle at each corner location

plt.imshow(photo_gray)  # Display the image with corners marked
plt.show()


'''

#Now using Canny Edge Detection and Contour Detection to find objects in an image

img = cv2.imread("/Users/jameswalter/Desktop/Photogammetry/Images/CUBE_1.JPG")

med_value = np.median(img)  # Calculate the median pixel value of the image

lower = int(max(0,0.7*med_value))  # Set lower threshold for edge detection to be 70% of median pixel value
upper = int(min(255,1.3*med_value))  # Set upper threshold for edge detection to be 130% of median pixel value

blurred_img = cv2.blur(img, ksize = (5,5))  # Blur the image to reduce noise 5,5 used for kernel size, adjust as needed

edges = cv2.Canny(image=blurred_img, 
                  threshold1=lower,
                  threshold2=upper) # Detect edges using Canny edge detector with specified thresholds, adding 250 to upper threshold for more sensitivity, adjust as needed

plt.imshow(edges)
plt.show()


