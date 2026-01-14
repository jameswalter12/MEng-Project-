## Update January 14th 2026

# Currently 

Current position is that I have a lot of time that needs to be spent on trying to work the resberry-pi cameras and equipment that I need to do the sensing and recording part of the 3D prints, it is a usb-c to micro-SD card, because I think this is what I need to access to actually initialise the cameras and then work the web-servers and things that I will need to actually control the cameras. 

In terms of Photogrammetry what I need to do is work on the actual combining of code data into the model - I STILL don't understand what the model is actually going to be.

Like what the hell are my parameters, how is this actually going to be developed into something. 

# As Tasked by Supervisor 

So before the Christmas break, my supervisor wanted me to work on printing out some railings for the 3D printer box for the small rasberry pi camera to move across. However, 

I am having a few doubts on how useful this will actually be, is it necessary for my project, is it even really a part of my project? from conversations had with Shehan, he was really struggling to get the camera to focus on the printed part (even while stationary) so I have some real doubts on the feasibility of this, or is the testing of feasability that my supervisor wants? 

Further, right at the end of the semester I was tasked with working with Hao, the PHD student to look at if we could attach an acoustic sensor to the print bed and see if we could get any interesting results. However, attaching the sensor caused the print bed to encounter some levelling issues that completely messed up the printers ability to actually print, and since then I have not been able to get it to print even after the acoustic sensor was removed. So this is a big thing that will likely need fixed. 

# Things figured out Recently 

So I discovered how to actually create the defect within the print, and it is literally just messing about with the print settings so it ACTUALLY makes a defect, I wish Shehan's code properly worked because I would be in such a good situation. I think his code is meant to be able to get an STL, slice the STL into the GCODE, provide defect creation optionality and print infill settings etc. All of that good stuff. This was all meant to be able to be done through an application, but when I tried to use it, it was extremely slow and buggy, probably need to look at this. When I was actually doing abit of printing all I was doing was using the PRUSA Slicer Application, which just did the slicing and GCODE stuff for me - much easier but I guess less connectivity and integration. 

## Next Steps and Things to do 

I definitely need to continue figuring out my Photogrammetry pipeline, I got to a point where I could do a low-level (LOW LEVEL) of image detection and it is included within this Github, but I did have quite a few issues and it is far from useable/completely ready for actual implementation within a final product. 

I need to be recording data ASAP is the big thing, my next big completion actually has to be connection and use of the camera sensors. Especially as there is likely to be image quality and focus issues. 

I also need to be wokring on Photogrammetry, my supervisor also recommended that I have a look at commercial tools and how they do it, this project is surely an investigation so I feel like if theres a commerciallly available tool there is no reason why I wouldn't be able to use this if it could be effective and useable with a digital twin. 

## Ideas 

Here is a thought, so in my previous code I was having some trouble with my photogrammetry code picking the correct candidate within a busy image, what I could potentially do, is use the expected cube, i.e. colours and shape, based on what STL I am feeding into the pipeline, if my code can access this, through the GCODE etc., the code can take the expected colour and maybe some parameters, to feed into a HSV detection sort of thing to narrow down some candidates. Even if the photo is taken at an early stage when it is very small, information on the colour will surely be helpful. This also means that the code can be completely adaptable to whatever shape/colour the target object will be because these parameters will be inputs throught the STL to GCODE process. Meaning I can use the code for (within reason) any object? 

# Caveats to the above 

This pipeline will be adapabtable to different parts, GCODE WILL give expected shape and position from the slicer - can parse, print bounding box, expected footprint, which can find candidates based on proximity etc... 

HOWEVER...This requires, image setup is controlled (fixed camera position), priors are provided (GCODE has no colour info so may have to be input). Wont be univeral if lighting changes etc. 
