# MEng-Project-
## Description

Will just try put some notes in here for now... 

Currently having a look at these sources: 

https://github.com/ataata107/Photogrammetry-OpenCV/tree/master  

https://github.com/worklifesg/Python-for-Computer-Vision-with-OpenCV-and-Deep-


## Github Rep 2 

Learning/blob/main/5.%20Object%20Detection%20with%20OpenCV%20and%20Python/1_ObjectDetection_OpenCV_Introduction_Template_Matching.ipynb 

* [] This source has information on some things about template matching with opencv, i.e. can send a photo, and a subset of that photo (smaller version) and then the code can search for that in the original... 

## Color Conversion

To do color conversion can easily jsut use: 

cv.cvtColor(input_image, flag) where flag determines the type of conversion - for greyscale use cv.COLOR_BGR2GRAY... 

## Object Color Tracking 

Can sometimes use color tracking to extract features - define a range for the colour blue in HSV first i.e. and upper and lower HEX bound. 

Then set a mask to only get blue colours (see link: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) 

## Calibration 

Do I have to do alot of this?? - If so how
