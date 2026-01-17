# Introduction

This file will provide in-depth documentation on what Shehan's code actually does, what it acheives, and what I can do with it... is it relevant to my project and should it be included in my work. 

# Main.py 

From what I can see this file brings all the other sub-files, all the other scripts, processes etc. and pulls it together in an application. Through examining this file, all other files will likely have to be investigated, debugged if necessary, and understood. 

* Class: Pythia(tk.Tk)
  So this class basically handles the creation of the whole application window and encompasses all its components and their respective function.


## def__init__(self) 

This function is the initialisation of the class, and all the lines below it will be run as the application is started (app = Pythia() is ran). Now all the functions within the Class, are called through inputs to the app, i.e. as the application is initialised, the tkinter functions, throught the Tkinter class (tk.Tk) allow application inputs to actually call the functions within the Class. So, the first few lines simply set the size and title and preset things, sets the icon, and initialises a few things: 

* Debug Mode & Debug Layer Comparison:

  So debug mode & Layer Comparison are enabled by default as the app is started, and there are a few things that are inialised. So the debug layers path is set as GCode/Output/defect_layers, and a default STL is loaded. Further a maximum of 12 debug layers are set. In a later function (slice_model()), if the sample STL is sliced, G-Code for this default STL is produced, the GCode is parsed, and layer coordinates are built. Model layers are also produced into a specified directory (line 42). Then if "print defect" is enabled, a defect G-Code is produced and saved. For debug comparison function generate_debug_layer_comparison() is used to turn that G-Code into defect layer images. When starting the print (start_print()), if layer comparison is enabled (via button in __init__), and debug_layer_comparison is true (which it is by default) a debug callback is set up, function setup_debug_layer_comparison() is called. It also calls function generate_debug_layer_images(). When debug_layer_comparison() is called, a pre-generated model image is loaded, and so are defect image. These are then compared through the function .compare(), which is from a seperately completed script compare_layer.py. Corrective G-Code is then sent if needed. The debug is capped at 12 layers.

* Printer Variables

  * The parser - The parser is initialised as None, but later initialised in function parse_gcode(). This is done via GCodeParser which is a function defined in the parser.py Python file.
   
  * Printer Connection - printer_connection is initially set as None, but is initialised in function connect() which uses PrinterConnection, a Class defined in the file connection.py.
 
  * Printer Control - printer_control is set as None, initialised in function connect(), and used PrinterControl, which is a class defined in control.py.
 
  * Connection Verificiation - is_connected is initially False, and isn't used anywhere else in the codes (no reads or writes after intiialisation). Potentially an unused placeholder.
 
  * Layer Coordinates - layers_coordinate is defined here as an empty list, which is filled in function parse_gocde() via parser.compute_layers(). compute_layers() is a function within parser.py that computes a list of per-ayer segments from G-Code moves.
 
  * Model Layer Path - model_layers_path is defined here and written to when exporting model layer images, in open_file(), if a G-Code file is opened directly and slice_model(), if an STL is sliced. This path is read from in update_plot() to display layer images in the UI, and in debug_layer_comparison_callback(), when the comparison needs model images. 

* Printer Settings - Printer settings such as filament diameter, nozzle diameter, layer height and flow modifier are set.

* Button Setting - Buttons/inputs are set for defect size, print toggles. A default directory for defect G-Code is set, and a path set initially at zero.

* 
  

    
