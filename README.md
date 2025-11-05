# TEST-FILES INFORMATION

## Description 

This file will contain all relevant notes and information for the contents of TEST-Files. 

## File 1 - OBJECT DETECTION TEST 

So this file is a preliminary attempt at some object detection using quite simple openCV packages, the code is hardly original with the majority of code being taken from source 1 (see below). Any image can be chosen for the code but it was designed with a simple photo of an iphone and works to some extent. However, as mentioned the code is not optimised at all for the photos and was more or less copied from the source (1). 

Source 1: https://github.com/worklifesg/Python-for-Computer-Vision-with-OpenCV-and-Deep-Learning/blob/main/5.%20Object%20Detection%20with%20OpenCV%20and%20Python/7_ObjectDetection_OpenCV_WaterShedAlgorithm.ipynb 

## File 2 - Edge/Corner Detection Testing 

So the good news is that the final peice of code in this file is a pretty simple method to detect edges and seems to work pretty well with the setup shown. However, the first two methods (both for edge detection) were very unsuccessfull, not sure why, but also not 100% sure how important edges actually are. Again source 1 was used for almost all of this... 

* NEXT STEP: Lets look at how this can actually be utilised and how I can start forming datasets from photos such as these, the upper angle does seem to be fairly successfull in the creation of the part.
* I should also look to see if I can isolate the simple part rather than collecting edges of the WHOLE thing as when we work on the photogrammetry we are not exactly going to be interested in the base or anything - maybe feature matching or something??
