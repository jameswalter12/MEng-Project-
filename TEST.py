import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    plt.show()

'''med_value = np.median(img)

lower = int(max(0,0.7*med_value))
upper = int(min(255,1.3*med_value))

blurred_img = cv2.blur(img, ksize = (6,6))

edges = cv2.Canny(img, threshold1=lower, threshold2= upper)
plt.imshow(edges) 
plt.axis('off')
plt.show()'''

img  = cv2.imread("/Users/jameswalter/Desktop/Photogammetry/Images/IMG_1668.JPG")

phone_blur = cv2.medianBlur(img, 25)
grey_phone = cv2.cvtColor(phone_blur, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey_phone, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)): 
    if hierarchy[0][i][3] == -1: 
        cv2.drawContours(img, contours, i, (255, 0, 0), 10)
display(img)


