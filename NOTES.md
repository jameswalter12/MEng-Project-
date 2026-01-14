# Notes

- [ ] January 14th 2026 - So there is a line of code in Shehan's main fail that directs the script to PRUSA slicer files that are downloaded within the src folder, however the code agent in VSCODE has clocked that these should be updated to be MacOS files, ensure this is changed

- [ ] So I need to look into what the serial ports are, there is functionality in the code that should let you connect to the 3D printer by the serial? Not sure how this works, if I have to do anything with the printer/code, if anything has to be initialised or sorted out - can I just put my laptop next to the printer?? Does anything have to actually be done here??
- [ ] With respect to the above item, it seems like I just need a usb connection with my laptop to the printer (so usb to usb-c) which I have? Well, I need usb to usb from printer to my dock, and then the dock to usb-c for my laptop...
- [ ] Currently, I think that all of the self.monitor_x_var and self.compare_mesh_var is currently not doing anything and is just defined and bound to the checkbox, self.compare_layer_img_var does however do something.

    * Now the above is an interesting one, the compare layer, is a function that allowes the comparison of  of a printed layer against the expected layer fromt the sliced model. So it takes a photo, saves it, then loads the model's layer passess both images to a layer comparison and computes a correction G-Code. Now if this works that is brilliant, however, surely this is where my Photogrammetry should come in, I really doubt this is done here... 

