import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from serial.tools import list_ports
import threading
import atexit
import time
from PIL import Image, ImageTk

from GCode import GCodeParser, DefectGenerator, VolumeMatrixGenerator
from Printer import PrinterConnection, PrinterControl
from Slicer import Slicer, SlicerConfig
from Monitor import Camera, CompareLayer

# This class handles the creation of the application window and its components
class Pythia(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Pythia')
        self.geometry('1280x720')

        # Set up the application icon
        self.icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pythia.png')
        self.icon_image = tk.PhotoImage(file=self.icon_path)
        self.tk.call('wm', 'iconphoto', self._w, self.icon_image)

        # Initialise debug and sample settings - these are used for testing without a physical printer
        self.debug_mode = True
        self.debug_layer_comparison = True
        self.debug_layer_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GCode/Output/defect_layers/')
        self.debug_end_layer = 12
        self.sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Samples/20 mm cube.stl')

        # Initialise printer-related variables and state
        self.parser = None
        self.printer_connection = None
        self.printer_control = None
        self.is_connected = False
        self.layers_coordinates = []
        self.model_layers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GCode/Output/model_layers/')

        # Set default printing parameters - should be checked 
        self.filament_diameter = 1.75
        self.nozzle_diameter = 0.4
        self.layer_height = 0.2
        self.flow_modifier = 0.95

        # Initialise monitoring toggles - this allows data to be collected during printing about various parameters
        self.compare_layer_img_var = tk.BooleanVar(value=True)
        self.compare_mesh_var = tk.BooleanVar(value=False)
        self.monitor_thermal_var = tk.BooleanVar(value=False)
        self.monitor_f_var = tk.BooleanVar(value=False)
        self.monitor_accel_var = tk.BooleanVar(value=False)

        # Initialise defect settings - allows intentional defects to be added to prints for testing purposes by altering the G-code 
        self.defect_type = None

        # Default defect settings - creates a 10 mm cube defect
        self.defect_size_var = tk.StringVar(value="10") # 10 mm
        self.print_defect_var = tk.BooleanVar(value=True)
        self.defect_gcode_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GCode/Output/defect_gcode/')
        self.defect_gcode_path = None

        self.is_paused = False

        # Set up slicer configuration paths
        self.config_default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NEWSLICER/WORKING_CONFIG.ini')
        self.custom_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Slicer/custom-profiles/custom_slicer_config.ini')

        # Initialise slicer configuration handler
        self.slicer_config = SlicerConfig(self.config_default_path, self.custom_config_path, self.log)

        # Define slicer configuration options which are the specific settings that can be adjusted in the GUI
        self.config_options = {
            'Layer Height': {
                'layer_height': ('Layer height', 'entry', None),
                'first_layer_height': ('First layer height', 'entry', None)
            },
            'Vertical Shells': {
                'perimeters': ('Perimeters', 'spinbox', (1, 20)),
                'spiral_vase': ('Spiral vase', 'checkbox', None)
            },
            # 'Horizontal Shells': {
            #     'solid_layers': ('Solid layers', 'spinbox', (1, 20)),
            #     'minimum_shell_thickness': ('Minimum shell thickness', 'entry', None)
            # },
            'Infill': {
                'fill_density': ('Fill density', 'entry', None),
                'fill_pattern': ('Fill pattern', 'combobox', ['alignedrectilinear', 'grid', 'honeycomb']),
                'top_fill_pattern': ('Top fill pattern', 'combobox', ['monotoniclines']),
                'bottom_fill_pattern': ('Bottom fill pattern', 'combobox', ['monotonic'])
            },
            'Skirt': {
                'skirts': ('Loops', 'spinbox', (0, 5)),
                'skirt_distance': ('Distance from brim/object', 'entry', None),
                'skirt_height': ('Skirt height', 'spinbox', (0, 5))
            },
            'Brim': {
                'brim_type': ('Brim type', 'combobox', ['outer_only', 'False'])
            },
            'Support Material': {
                'support_material': ('Generate support material', 'checkbox', None)
            }#,
            #'Misc': {
            #    'arc_fitting': ('Arc fitting', 'checkbox', None)
            #}
        }

        # Initialise model, config, and output paths
        self.model_path = None
        self.config_path = None

        # This is the path to save the generated G-code files
        self.output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Slicer/output_gcode')
       
        # This is the path to the slicer executable which is the program that converts 3D models into G-code - Note: Update version as needed to macOS since files are in Windows

        self.slicer_path = '/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer'

        # This is the path to store downloaded images from cameras during monitoring
        self.download_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Monitor/Photos/')
        os.makedirs(self.download_base_dir, exist_ok=True)
        
        # Thermal camera URL must be set during camera setup
        self.thermal_camera_url = None

        # Initialise the user interface
        self.initUI()
        atexit.register(self.cleanup_on_exit)
    

    def cleanup_on_exit(self):
        """Function to clean up resources when the app exits"""
        if self.printer_connection:
            self.printer_connection.disconnect()
            print("Disconnected from printer")

    # This function simply initialises the user interface of the application
    def initUI(self):
        """Initialise user interface"""
        # Create the Notebook
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)

        # Tab 1: Control
        control_tab = tk.Frame(notebook)
        notebook.add(control_tab, text='Control')

        # Tab 3: Slicer Settings
        settings_tab = tk.Frame(notebook)
        notebook.add(settings_tab, text='Slicer Settings')

        # Set up the tabs
        self.setupControlTab(control_tab)
        self.setupConfigTab(settings_tab)

        # Configure styles
        self.style = ttk.Style(self)
        self.style.configure('Switch.TCheckbutton', indicatorsize=25)

        # If in debug mode, load sample model and slice it, debug mode allows for testing without a physical printer
        if self.debug_mode:
            self.model_path = self.sample_path
            self.slice_model()

    # This function sets up the control tab of the application
    def setupControlTab(self, frame):
        # Main layout frames
        control_frame = tk.Frame(frame, padx=5, pady=5, borderwidth=2, relief="groove")
        control_frame.grid(row=0, column=0, sticky='nsew')

        monitor_frame = tk.Frame(frame, padx=5, pady=5)
        monitor_frame.grid(row=0, column=1, sticky='nsew')

        # Control Frame Components - printer connection is handled in the seperate scripts under Printer etc. 
        port_frame = tk.Frame(control_frame)
        port_frame.grid(row=0, column=0, sticky='ew')
        self.port_label = ttk.Label(port_frame, text='Select Port')
        self.port_label.grid(row=0, column=0, sticky='ew')
        # The line below lists available serial ports for printer connection the list comes from the serial.tools.list_ports module which is in the pyserial package
        self.port_selector = ttk.Combobox(port_frame, values=[port.device for port in list_ports.comports()])
        self.port_selector.grid(row=1, column=0, sticky='ew')

        # This just makes a refresh button to update the list of available serial ports
        self.refresh_ports_button = tk.Button(port_frame, text='Refresh Ports', command=self.refresh_ports)
        self.refresh_ports_button.grid(row=2, column=0, pady=5, sticky='ew')

        # Baudrate selection - Baudrate is the speed of communication over the serial connection - 115200 is a common default for 3D printers. 
        baudrate_frame = tk.Frame(control_frame)
        baudrate_frame.grid(row=1, column=0, sticky='ew')
        self.baudrate_label = ttk.Label(baudrate_frame, text='Select Baudrate')
        self.baudrate_label.grid(row=0, column=0, sticky='ew')
        self.baudrate_selector = ttk.Combobox(baudrate_frame, values=['115200'])
        self.baudrate_selector.current(0)
        self.baudrate_selector.grid(row=1, column=0, sticky='ew')

        # Connection button - connects to the printer using the selected port and baudrate
        self.connection_button = tk.Button(control_frame, text='Connect', command=self.connect)
        self.connection_button.grid(row=2, column=0, pady=5, sticky='ew')
        self.upload_button = tk.Button(control_frame, text='Upload Model', command=self.open_file)
        self.upload_button.grid(row=5, column=0, pady=5, sticky='ew')
        self.slice_button = tk.Button(control_frame, text='Slice Model', command=self.slice_model)
        self.slice_button.grid(row=6, column=0, pady=5, sticky='ew')
    
        # Correction frame is for setting up monitoring and correction options during printing
        correction_frame = tk.Frame(control_frame, borderwidth=2, relief="groove")
        correction_frame.grid(row=3, column=0, pady=10, sticky='ew')

        # This toggle allows toggling of layer image comparison during printing
        self.compare_layer_img_toggle = ttk.Checkbutton(correction_frame, text='Compare Layer Image', variable=self.compare_layer_img_var)
        self.compare_layer_img_toggle.grid(row=0, column=0, pady=5, sticky='ew')

        # Makes a toggle for mesh comparison during printing
        self.compare_mesh_toggle = ttk.Checkbutton(correction_frame, text='Compare Mesh', variable=self.compare_mesh_var)
        self.compare_mesh_toggle.grid(row=1, column=0, pady=5, sticky='ew')

        # This toggle allows monitoring of the thermal camera during printing
        self.monitor_thermal_toggle = ttk.Checkbutton(correction_frame, text='Use Thermal Camera', variable=self.monitor_thermal_var)

        self.monitor_f_toggle = ttk.Checkbutton(correction_frame, text='Use Filament Sensor', variable=self.monitor_f_var)
        self.monitor_f_toggle.grid(row=3, column=0, pady=5, sticky='ew')

        self.monitor_accel_toggle = ttk.Checkbutton(correction_frame, text='Use Accelerometer', variable=self.monitor_accel_var)
        self.monitor_accel_toggle.grid(row=4, column=0, pady=5, sticky='ew')

        defect_frame = tk.Frame(control_frame, borderwidth=2, relief="groove")
        defect_frame.grid(row=4, column=0, pady=10, sticky='ew')

        defect_type_label = ttk.Label(defect_frame, text='Select Defect Type')
        defect_type_label.grid(row=0, column=0, pady=5, sticky='ew')
        self.defect_type_entry = ttk.Combobox(defect_frame, values=['Cube'])
        self.defect_type_entry.current(0)
        self.defect_type_entry.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        defect_size_label = ttk.Label(defect_frame, text='Select Defect Size')
        defect_size_label.grid(row=2, column=0, pady=5, sticky='ew')
        self.defect_size_selector = ttk.Spinbox(defect_frame, from_=1, to=90, textvariable=self.defect_size_var, increment=2)
        self.defect_size_selector.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        self.create_defect_switch = ttk.Checkbutton(defect_frame, text='Print Defect', variable=self.print_defect_var, style='Switch.TCheckbutton')
        self.create_defect_switch.grid(row=4, column=0, padx=5, pady=5, sticky='ew')

        self.camera_setup = tk.Button(control_frame, text='Setup Cameras', command=self.setup_cameras)
        self.camera_setup.grid(row=7, column=0, pady=5, sticky='ew')
        self.test_button = tk.Button(control_frame, text='Test', command=self.correction_test)
        self.test_button.grid(row=8, column=0, pady=5, sticky='ew')

        self.print_button = tk.Button(control_frame, text='Print', command=self.start_print)
        self.print_button.grid(row=9, column=0, pady=5, sticky='ew')
        self.pause_button = tk.Button(control_frame, text='Pause', command=self.pause_resume_print)
        self.pause_button.grid(row=10, column=0, pady=5, sticky='ew')
        self.cancel_button = tk.Button(control_frame, text='Cancel', command=self.cancel_print)
        self.cancel_button.grid(row=11, column=0, pady=5, sticky='ew')

        viewer_frame = tk.Frame(monitor_frame, borderwidth=2, relief="groove")
        viewer_frame.grid(row=0, column=0, sticky='nsew')

        terminal_frame = tk.Frame(monitor_frame, borderwidth=2, relief="groove")
        terminal_frame.grid(row=1, column=0, sticky='nsew', pady=10)  # Move terminal below viewer

        camera_frame = tk.Frame(monitor_frame, borderwidth=2, relief="groove", padx=10, pady=10)
        camera_frame.grid(row=0, column=1, rowspan=2, sticky='nsew', padx=10)  # Move cameras to right side, spanning both rows

        #monitor_frame.grid_rowconfigure(0, weight=2)  # Give more weight to viewer and terminal
        #monitor_frame.grid_rowconfigure(1, weight=1)  # Less weight to cameras for layout balance
        #monitor_frame.grid_columnconfigure([0, 1], weight=1)

        # Viewer setup
        self.canvas = PlotCanvas(viewer_frame)
        self.canvas.frame.grid(row=0, column=0, sticky='nw')

        slider_frame = tk.Frame(viewer_frame)
        slider_frame.grid(row=0, column=2, rowspan=2, sticky='ew')

        # Slider for layers
        self.slider = tk.Scale(slider_frame, from_=0, to=100, orient='vertical', command=self.update_plot, length=360)
        self.slider.pack(fill='both', expand=True)

        #viewer_frame.grid_rowconfigure(0, weight=1)
        #viewer_frame.grid_columnconfigure(0, weight=1)

        # Terminal setup
        self.terminal = tk.Text(terminal_frame, height=15, state='disabled')
        self.terminal.grid(row=0, column=0, sticky='nsew', pady=5)
        terminal_frame.grid_rowconfigure(0, weight=1)
        terminal_frame.grid_columnconfigure(0, weight=1)

        command_frame = tk.Frame(terminal_frame)
        command_frame.grid(row=1, column=0, sticky='nsew')
        self.command_input = ttk.Entry(command_frame, width=80)
        self.command_input.grid(row=0, column=0, sticky='nsew')
        self.command_input.bind('<Return>', self.send_command)
        send_button = ttk.Button(command_frame, text="Send", command=self.send_command)
        send_button.grid(row=0, column=1, sticky='nsew')

        # Camera setup ensuring it is in row 1, spanning both columns
        camera_label = tk.Label(camera_frame, text=f'Camera {1}: [Not Connected]', bg='grey', fg='white', width=30, height=15)
        camera_label.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

        camera_label = tk.Label(camera_frame, text=f'Camera {2}: [Not Connected]', bg='grey', fg='white', width=50, height=10)
        camera_label.grid(row=2, column=0, sticky='ew', padx=5, pady=5)

        self.thermal_cam_label = tk.Label(camera_frame, text='Thermal Camera: [Not Connected]', bg='grey', fg='white', width=50, height=16)
        self.thermal_cam_label.grid(row=4, column=0, sticky='ew', padx=5, pady=5)

    def setupConfigTab(self, frame):
        num_columns = 3  # Number of columns for the sections

        # Use a main container frame to hold all the sections
        container = ttk.Frame(frame)
        container.grid(row=0, column=0, sticky='nsew')
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Mapping of settings to sections as defined in self.config_options
        section_index = 0
        for section, settings in self.config_options.items():
            row_index = section_index // num_columns
            col_index = section_index % num_columns

            sec_frame = ttk.LabelFrame(container, text=section, padding="10 10 10 10")
            sec_frame.grid(row=row_index, column=col_index, sticky='nsew', padx=10, pady=5)
            container.grid_columnconfigure(col_index, weight=1)

            # Allow the sec_frame to grow with the content
            sec_frame.grid_rowconfigure(row_index, weight=1)
            sec_frame.grid_columnconfigure(0, weight=1)
            sec_frame.grid_columnconfigure(1, weight=1)

            row_count = 0
            for key, (label, widget_type, options) in settings.items():
                value = self.slicer_config.get_setting(key)

                # Place label
                label_widget = ttk.Label(sec_frame, text=label)
                label_widget.grid(row=row_count, column=0, sticky='w', padx=5, pady=2)

                # Place appropriate widget with current value set
                if widget_type == 'combobox':
                    widget = ttk.Combobox(sec_frame, values=options, width=20)
                    widget.set(value)  # Set current value
                elif widget_type == 'spinbox':
                    widget = ttk.Spinbox(sec_frame, from_=options[0], to=options[1], width=20)
                    widget.set(value)  # Set current value
                elif widget_type == 'checkbox':
                    var = tk.IntVar(value=int(value) if value.isdigit() else 0)
                    widget = tk.Checkbutton(sec_frame, variable=var)
                elif widget_type == 'entry':
                    widget = ttk.Entry(sec_frame, width=20)
                    widget.insert(0, value)

                widget.grid(row=row_count, column=1, sticky='ew', padx=5, pady=2)
                widget.bind("<FocusOut>", lambda e, k=key, w=widget: self.update_config_setting(k, w.get()))

                row_count += 1

            section_index += 1

        # Save button in a separate frame at the bottom
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=10)
        frame.grid_rowconfigure(1, weight=0)  # Ensures the button frame does not expand
        save_button = ttk.Button(button_frame, text='Save Configuration', command=self.save_config_settings)
        save_button.pack(fill='x', expand=True)

    def setup_cameras(self):
        # Wizard window
        wizard_window = tk.Toplevel(self)
        wizard_window.title("Camera Setup Wizard")
        wizard_window.geometry("350x450")

        # Number of cameras
        tk.Label(wizard_window, text="Enter number of cameras:").pack()
        num_cameras_spinbox = tk.Spinbox(wizard_window, from_=1, to=10, width=5)
        num_cameras_spinbox.pack()

        # Checkbox and URL entry for thermal camera
        thermal_cam_var = tk.IntVar()
        thermal_cam_checkbox = tk.Checkbutton(wizard_window, text="Include thermal camera", variable=thermal_cam_var)
        thermal_cam_checkbox.pack()
        
        tk.Label(wizard_window, text="Thermal Camera URL:").pack()
        thermal_cam_url_entry = tk.Entry(wizard_window)
        thermal_cam_url_entry.pack()

        # Frame for URL entries
        url_frame = tk.Frame(wizard_window)
        url_frame.pack()
        url_entries = []

        def create_url_entries():
            nonlocal url_entries
            for widget in url_frame.winfo_children():
                widget.destroy()
            num_cameras = int(num_cameras_spinbox.get())
            url_entries = []
            for i in range(num_cameras):
                tk.Label(url_frame, text=f"URL for Camera {i+1}:").pack()
                url_entry = tk.Entry(url_frame)
                url_entry.pack()
                url_entries.append(url_entry)
            return url_entries

        # Button to confirm number and generate URL entry fields
        tk.Button(wizard_window, text="Set Cameras", command=create_url_entries).pack()

        def setup_thermal_camera(url):
            self.thermal_camera_url = url
            self.thermal_camera_stream()

        def finalize_setup():
            url_addresses = [entry.get().strip() for entry in url_entries]
            url_addresses = [u for u in url_addresses if u]
            normal_cameras = [Camera(url, self.download_base_dir, self.log) for url in url_addresses]  # Create Camera objects for each URL
            for camera in normal_cameras:
                camera.test_camera()

            if thermal_cam_var.get() and thermal_cam_url_entry.get():
                setup_thermal_camera(thermal_cam_url_entry.get())  # Set up thermal camera separately

        # Finalize setup button
        tk.Button(wizard_window, text="Finish Setup", command=finalize_setup).pack()

    def refresh_ports(self):
        """Refresh the list of available serial ports."""
        self.port_selector['values'] = [port.device for port in list_ports.comports()]
        if self.port_selector['values']:
            self.port_selector.current(0)

    def update_config_setting(self, key, value):
        self.slicer_config.update_setting(key, value)

    def save_config_settings(self):
        self.slicer_config.save_settings()
        self.config_path = self.custom_config_path

    def log(self, message):
        try:
            self.terminal.config(state='normal')  # Temporarily make it normal to insert text
            self.terminal.insert(tk.END, message + '\n')
            self.terminal.see(tk.END)  # Scroll to the bottom of the log
            self.terminal.config(state='disabled')  # Set it back to disabled
        except tk.TclError as e:
            print("Failed to log message in terminal widget:", e)

    def open_file(self):
        self.model_path = filedialog.askopenfilename(title="Open Model File", filetypes=(("CAD Files", "*.stl"), ("G-Code Files", "*.gcode"), ("All Files", "*.*")))
        if self.model_path:
            file_extension = os.path.splitext(self.model_path)[-1].lower()
            if file_extension == '.gcode':
                self.log(f"Opened G-Code file: {self.model_path}")
                self.parse_gcode(self.model_path)
                VolumeMatrixGenerator(g_path=self.model_path, logger=self.log).export_images(dir=self.model_layers_path)
                if self.print_defect_var.get():
                    size = int(self.defect_size_var.get()) / 2
                    self.defect_gcode_path = DefectGenerator(self.model_path, self.log).run(size, -size, size, -size, self.filament_diameter, self.defect_gcode_dir)
            elif file_extension == '.stl':
                self.log(f"Opened STL file: {self.model_path}")
            else:
                self.log("Unsupported file type opened.")

    def parse_gcode(self, file_path):
        self.log(f"Parsing GCode file: {file_path}")
        self.parser = GCodeParser(file_path, self.log)
        _ = self.parser.parse('enhanced')
        self.layers_coordinates = self.parser.compute_layers()
        self.xmin, self.xmax, self.ymin, self.ymax, _, _, _ = self.parser.describe()
        self.update_slider_range()
    
    def slice_model(self):
        def threaded_slice():
            if self.model_path and self.model_path.endswith('.stl'):
                if not self.config_path:
                    self.config_path = self.config_default_path

                model_name = os.path.splitext(os.path.basename(self.model_path))[0] + '.gcode'
                gcode_path = os.path.join(self.output_path, model_name)

                self.log(f"[slicer] using config: {self.config_path}")
                slicer = Slicer(self.config_path, self.model_path, gcode_path, self.slicer_path, self.log)
                slicer.generate_gcode()

                self.parse_gcode(gcode_path)
                self.log("Slicing completed successfully.")  # Confirm completion

                VolumeMatrixGenerator(g_path=gcode_path, logger=self.log).export_images(dir=self.model_layers_path)

                if self.print_defect_var.get():
                    size = int(self.defect_size_var.get()) / 2
                    self.defect_gcode_path = DefectGenerator(gcode_path, self.log).run(size, -size, size, -size, self.filament_diameter, self.defect_gcode_dir)
            else:
                self.log("No STL file is loaded or selected file is not an STL.")
                return  # Exit the thread if not an STL file

        slice_thread = threading.Thread(target=threaded_slice)
        slice_thread.daemon = True  # Ensures the thread does not prevent the program from exiting
        slice_thread.start()

    def update_slider_range(self):
        if self.layers_coordinates:
            self.slider.config(from_=0, to=len(self.layers_coordinates) - 1)
            self.slider.set(1)
            self.update_plot(1)

    def update_plot(self, value):
        layer_index = int(value)
        if self.layers_coordinates and layer_index < len(self.layers_coordinates):
            self.canvas.plot_layers(self.layers_coordinates, layer_index, self.xmin, self.xmax, self.ymin, self.ymax)
            self.canvas.display_image(self.model_layers_path, layer_index)

    def connect(self):
        def threaded_connect():
            if self.port_selector.get() and self.baudrate_selector.get():
                selected_port = self.port_selector.get()
                selected_baudrate = int(self.baudrate_selector.get())
                self.printer_connection = PrinterConnection(selected_port, selected_baudrate, self.log)
                self.printer_control = PrinterControl(self.printer_connection, self.log)
                self.printer_connection.connect()
                if self.debug_mode:
                    self.printer_control.set_debug_end_layer(self.debug_end_layer)
            elif self.port_selector.get():
                messagebox.showerror("Error", "Please select a baudrate")
            elif self.baudrate_selector.get():
                messagebox.showerror("Error", "Please select a port")
            else:
                messagebox.showerror("Error", "Please select a port and baudrate")

        # Create and start the thread
        connect_thread = threading.Thread(target=threaded_connect)
        connect_thread.daemon = True
        connect_thread.start()

    def send_command(self):
        command = self.command_input.get().strip()  # Get the command from the input field
        if self.printer_connection:
            self.printer_control.enqueue_command(command, to_front=True)
            self.command_input.delete(0, 'end')  # Clear the input field after sending
        else:
            messagebox.showerror("Error", "Printer not connected. Please connect to a printer first.")

    # def start_print(self):
    #     if self.printer_control and self.parser:
    #         self.log("Starting print...")
    #         if self.print_defect_var.get():
    #             self.printer_control.enqueue_dict_commands(GCodeParser(self.defect_gcode_path, self.log).parse('basic'))
    #         else:
    #             self.printer_control.enqueue_dict_commands(self.parser.get_commands())
    #     else:
    #         messagebox.showerror("Error", "Ensure the printer is connected and GCode is loaded.")

    def start_print(self):
        if self.printer_control and self.parser:
            self.log("Starting print...")
            if self.print_defect_var.get():
                if not self.defect_gcode_path:
                    self.log("Defect print requested but defect G-code not available. Falling back to original G-code.")
                    gcode_commands = self.parser.get_commands()
                else:
                    gcode_commands = GCodeParser(self.defect_gcode_path, self.log).parse('basic')
            else:
                gcode_commands = self.parser.get_commands()
            
            if gcode_commands is None:
                self.log("Error: Failed to parse G-code commands.")
                messagebox.showerror("Error", "Failed to parse G-code commands. Check the log for details.")
                return

            # Set up layer comparison if enabled
            if self.compare_layer_img_var.get():
                if self.debug_layer_comparison:
                    if not self.defect_gcode_path:
                        self.log("Debug layer comparison disabled: no defect G-code generated.")
                        self.printer_control.set_compare_layer_img_toggle(False)
                    else:
                        self.setup_debug_layer_comparison()
                        self.printer_control.set_compare_layer_img_toggle(True)
                        self.printer_control.set_gcode_parser(self.parser)
                else:
                    self.setup_layer_comparison()
                    self.printer_control.set_compare_layer_img_toggle(True)
                    self.printer_control.set_gcode_parser(self.parser)
            else:
                self.printer_control.set_compare_layer_img_toggle(False)
            
            # Set debug mode and end layer
            self.printer_control.set_debug_mode(self.debug_mode)
            self.printer_control.set_debug_end_layer(self.debug_end_layer)
            
            # Generate model layer images if in debug mode
            if self.debug_layer_comparison and self.defect_gcode_path:
                self.generate_debug_layer_images()
            
            # Send all commands to the printer
            try:
                self.printer_control.enqueue_dict_commands(gcode_commands)
                self.log("Print started successfully.")
            except Exception as e:
                self.log(f"Error starting print: {str(e)}")
                messagebox.showerror("Error", f"Failed to start print: {str(e)}")
        else:
            messagebox.showerror("Error", "Ensure the printer is connected and G-code is loaded.")

    def pause_resume_print(self):
        if self.printer_control:
            if self.is_paused:
                self.is_paused = False
                self.printer_control.resume_print()
            else:
                self.is_paused = True
                self.printer_control.pause_print()
        else:
            messagebox.showerror("Error", "Ensure printer is connected and print has started")

    def cancel_print(self):
        if self.printer_control:
            self.printer_control.cancel_print()
        else:
            messagebox.showerror("Error", "Ensure printer is connected and print has started")

    def setup_layer_comparison(self):
        if not hasattr(self, 'camera') or not self.camera:
            messagebox.showerror("Error", "Camera not set up. Please set up a camera before starting the print.")
            return

        self.compare_layer = CompareLayer(self.nozzle_diameter, self.filament_diameter, self.layer_height)
        self.printer_control.set_layer_callback(self.layer_comparison_callback)
    
    def layer_comparison_callback(self, current_layer):
        self.log(f"Processing layer {current_layer}")
        
        # Skip comparison for layer 0
        if current_layer == 0:
            self.log("Skipping comparison for initial layer")
            return
        
        # Capture image using the camera
        image = self.camera.capture_image()
        if image:
            # Save the captured image
            captured_image_path = os.path.join(self.download_base_dir, f'captured_layer_{current_layer}.jpg')
            image.save(captured_image_path)
            
            # Get the model image path for the current layer
            model_image_path = os.path.join(self.model_layers_path, f'{current_layer:03d}.tiff')
            
            # Get the current extrusion width from PrinterControl
            current_width = self.printer_control.get_current_width()
            
            # Use CompareLayer to get correction G-code
            correction_gcode = self.compare_layer.compare(model_image_path, captured_image_path, current_width, self.flow_modifier)
            
            # If correction G-code is generated, execute it
            if correction_gcode:
                self.log("Executing correction G-code")
                for command in correction_gcode:
                    self.printer_control.enqueue_command(command, to_front=True)

    def setup_debug_layer_comparison(self):
        self.compare_layer = CompareLayer(self.nozzle_diameter, self.filament_diameter, self.layer_height)
        self.printer_control.set_layer_callback(self.debug_layer_comparison_callback)
        self.printer_control.set_debug_end_layer(self.debug_end_layer)

    def generate_debug_layer_images(self):
        # Use VolumeMatrixGenerator to create defect layer images
        VolumeMatrixGenerator(g_path=self.defect_gcode_path, logger=self.log).export_images(dir=self.debug_layer_img_dir)

    def debug_layer_comparison_callback(self, current_layer):
        self.log(f"Processing layer {current_layer}")
        print(f"Processing layer {current_layer}")
        
        # Skip comparison for layer 0
        if current_layer == 0:
            self.log("Skipping comparison for initial layer")
            return
        
        # Use the pre-generated image as both the model and the "captured" image
        model_image_path = os.path.join(self.model_layers_path, f'{current_layer:03d}.tiff')
        actual_image_path = os.path.join(self.debug_layer_img_dir, f'{current_layer:03d}.tiff')
        
        # Get the current extrusion width from PrinterControl
        current_width = self.printer_control.get_current_width()
        
        # In debug mode, we use the same image for both model and "captured"
        correction_gcode = self.compare_layer.compare(model_image_path, actual_image_path, current_width, self.flow_modifier, True)
        
        if correction_gcode:
            self.log("Executing correction G-code")
            # self.printer_control.enqueue_commands(correction_gcode, to_front=True)
            for cmd in correction_gcode:
                self.printer_control.send_command(cmd)
        print(f"Processing layer {current_layer} done")

    def thermal_camera_stream(self):
        pass

    def correction_test(self):
        def correction_test_thread():
            print("pausing print")
            self.printer_control.pause_print()
            print("print pause commands sent, waiting for ready status")
            
            # Wait for the printer to signal that it has actually paused
            self.printer_control.pause_complete.wait()

            cube_correction_test_gcode = [
                    "CRT: G1 X0 Y0 ; Move to the starting point",
                    "CRT: G1 X50 Y0 ; Draw the first side (right)",
                    "CRT: G1 X50 Y50 ; Draw the second side (up)",
                    "CRT: G1 X0 Y50 ; Draw the third side (left)",
                    "CRT: G1 X0 Y0 ; Draw the fourth side (down)",
                    "!RESUME"
                ]
            
            print("correcting")
            self.printer_control.enqueue_commands(cube_correction_test_gcode, to_front=True)
        
        correction_thread = threading.Thread(target=correction_test_thread)
        correction_thread.daemon = True
        correction_thread.start()

class PlotCanvas(FigureCanvasTkAgg):
    def __init__(self, parent):
        self.fig = Figure(figsize=(6, 3))  # Adjusted size for two plots side by side
        self.ax_plot = self.fig.add_subplot(121)
        self.ax_image = self.fig.add_subplot(122)
        super().__init__(self.fig, master=parent)

        # Create a frame to hold the canvas and toolbar
        self.frame = tk.Frame(parent)
        self.frame.grid(sticky='ew')

        # Create and add the toolbar
        self.toolbar = NavigationToolbar2Tk(self, self.frame)
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky='ew')

        # Add the canvas to the frame
        self.get_tk_widget().grid(row=1, column=0, sticky='ew')

    def plot_layers(self, layers_coordinates, layer_index, xmin, xmax, ymin, ymax):
        # Plot layer updates
        self.ax_plot.clear()
        layer = layers_coordinates[layer_index]
        for segment in layer:
            x_values = [point['x'] for point in segment]
            y_values = [point['y'] for point in segment]
            linewidth = segment[0]['width'] if ('width' in segment[0]) else 1
            self.ax_plot.plot(x_values, y_values, linewidth=linewidth * 6)
        
        self.ax_plot.set_title(f'Layer {layer_index}')
        self.ax_plot.axis('scaled')
        self.ax_plot.set_xlim(round(xmin - (xmin * 0.1)), round(xmax + (xmax * 0.1)))
        self.ax_plot.set_ylim(round(ymin - (ymin * 0.1)), round(ymax + (ymax * 0.1)))
        self.draw()

    def display_image(self, image_dir, layer_index):
        image_path = os.path.join(image_dir, f'{layer_index:03d}.tiff')
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = (img.resize((250, 250), Image.LANCZOS)).transpose(Image.FLIP_LEFT_RIGHT)
            self.photo_image = ImageTk.PhotoImage(img)
            
            self.ax_image.clear()
            self.ax_image.imshow(img)
            self.ax_image.axis('off')  # Hide axes for images
            self.draw()
        else:
            self.ax_image.clear()
            self.ax_image.text(0.5, 0.5, 'No image available', ha='center', va='center')
            self.ax_image.axis('off')
            self.draw()

app = Pythia()
app.mainloop()