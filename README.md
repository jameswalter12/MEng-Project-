# MEng-Project-
## Description

Will just try put some notes in here for now... 

Currently having a look at this source: https://github.com/ataata107/Photogrammetry-OpenCV/tree/master 

# Color Conversion

To do color conversion can easily jsut use: 

cv.cvtColor(input_image, flag) where flag determines the type of conversion - for greyscale use cv.COLOR_BGR2GRAY... 

# Object Color Tracking 

Can sometimes use color tracking to extract features - define a range for the colour blue in HSV first i.e. and upper and lower HEX bound. 

Then set a mask to only get blue colours (see link: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) 

# Calibration 

Do I have to do alot of this?? - If so how
