#!/usr/bin/env python3
# GUI.py
# contains all the functions and classes needed to run the GUI interface for MSIGen
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pathlib import Path
import os
import win32api
from threading import Thread
from time import time
from copy import deepcopy

from MSIGen import msigen
from MSIGen import visualization as vis


def verify_rawfile_names_gui(rawfile_paths):
    """Ensures that all file names that are selected in the GUI:
    1: All have the same path
    2: All have the same file extension
    3: All have the same file name, apart from a final number
    4: Contain a unique number at the end of the file name
    
    Input: list(str) of file paths
    outputs:
        rawfile_paths: A single file path as a string if only one path is given. Otherwise, this is the same as the input.
        filenames_checked: bool
    """
    filenames_checked = False
    print(rawfile_paths)

    # check that data was selected
    if len(rawfile_paths) == 0:
        error_message = "No raw data files were selected!"
        tk.messagebox.showerror("Select file error", error_message)
    elif len(rawfile_paths) == 1:
        if rawfile_paths[0]=='':
            error_message = "No raw data files were selected!"
            tk.messagebox.showerror("Select file error", error_message)
        # if only one file, return a string instead of a list
        else:
            if Path(rawfile_paths[0]).exists():
                rawfile_paths = rawfile_paths[0]
                filenames_checked = True
            else:
                error_message = "The data file given does not exist."
                tk.messagebox.showerror("Select file error", error_message)


    elif len(rawfile_paths) > 1:
            # Check all have same file extension
            if len(set([os.path.splitext(i)[1] for i in rawfile_paths])) != 1:
                error_message = "Not all selected files had the same file extension!"
                tk.messagebox.showerror("Select file error", error_message)
            
            # check filenames only differ by number at the end
            else:
                namebodies = []
                names = set([os.path.splitext(i)[0] for i in rawfile_paths])

                for name in names:
                    iterator = 0
                    for i in name[::-1]:
                        if i.isdigit():
                            iterator+=1
                        else:
                            break
                    
                    if iterator<1:
                        error_message = "All selected files must differ by a number at the end of the file name."
                        tk.messagebox.showerror("Select file error", error_message)
                    namebodies.append(name[:-iterator])
                    
                if len(set(namebodies)) != 1:
                    error_message = "All selected files must only differ by a number at the end of the file name."
                    tk.messagebox.showerror("Select file error", error_message)
                
                else:
                    if all([Path(i).exists() for i in rawfile_paths]):
                        filenames_checked = True
                    else:
                        error_message = "At least one data file given does not exist."
                        tk.messagebox.showerror("Select file error", error_message)
    return rawfile_paths, filenames_checked


def get_download_path():
    """Returns the default downloads path for linux or windows"""
    try:
        if os.name == 'nt':
            import winreg
            sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
            downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
                location = winreg.QueryValueEx(key, downloads_guid)[0]
            return location
        else:
            if os.path.exists(os.path.join(os.path.expanduser('~'), 'Downloads')):
                return os.path.join(os.path.expanduser('~'), 'Downloads')
    except: 
        return ''

def get_final_mass_list_gui(metadata):
    """Gets the mass list in displayable form for the GUI"""
    mass_list = deepcopy(metadata['final_mass_list'])
    output_table = []
    
    if metadata['is_MS2']:
        columns = ["Index", "Precursor m/z", "Fragment m/z"]
        for i in range(1,len(mass_list)):
            # add in fragment if needed
            if len(mass_list[i])==3:
                output_table.append([i, mass_list[i][0], ''])

            elif len(mass_list[i])==4:
                output_table.append([i, mass_list[i][0], mass_list[i][1]])
    
    else:
        columns = ["Index", "m/z"]
        for i in range(1,len(mass_list)):
            output_table.append([i, mass_list[i][0]])

    if metadata['is_mobility']:
        columns.append("Mobility")
        for i in range(1,len(mass_list)):
            if mass_list[i][-2]==0:
            # add in fragment if needed
                output_table[i-1].append('')
            else:
                output_table[i-1].append(mass_list[i][-2])

    if any([i[-1] != 0 for i in mass_list][1:]):
        columns.append("Polarity")
        for i in range(1,len(mass_list)):
        # convert polarity to symbol
            pol = mass_list[i][-1]
            if pol > 0:
                output_table[i-1].append('+')
            elif pol < 0:
                output_table[i-1].append('-')
            elif pol == 0:
                output_table[i-1].append('')

    return columns, output_table

class MyButton(tk.Button):
    """Button that can be selected with Tab and pressed with Return"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Return>', lambda event: self.invoke())

class MasterWindow(tk.Tk):
    """The main window of MSIGen. 
Files and parameters are input here before running the data extraction workflow."""
    def __init__(self):
        super().__init__()
        self.title("MSI Generator")
        self.geometry("600x600")  # Set size for the control window

        self.protocol("WM_DELETE_WINDOW", self.destroy_all_windows)
        
        self.rawfile_paths = tk.StringVar(value="")
        self.mass_list_path = tk.StringVar(value="")
        self.output_file_path = tk.StringVar(value="")
        self.img_h = tk.StringVar(value="10")
        self.img_w = tk.StringVar(value="10")
        self.is_MS2_var = tk.IntVar()
        self.is_mob_var = tk.IntVar()
        self.scale = tk.DoubleVar(value=100)
        self.threshold = tk.DoubleVar(value=0.)

        # left frame
        self.rawfiles_frame = tk.Frame(self)
        self.rawfiles_frame.pack(side = tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        # selecting raw files
        self.raw_files_label = tk.Label(self.rawfiles_frame, text = 'MS data files to use:')
        self.raw_files_label.pack(side=tk.TOP, anchor = tk.W)
        self.rawfiles_box = tk.Listbox(self.rawfiles_frame, selectmode=tk.EXTENDED)
        self.rawfiles_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.button_frame = tk.Frame(self.rawfiles_frame)
        self.button_frame.pack(anchor=tk.CENTER)
        self.open_explorer_button = MyButton(self.button_frame, text="Open New Files", command=self.open_file_explorer,width=12, height=1)
        self.open_explorer_button.pack(side=tk.LEFT, padx=1, pady=5)
        self.delete_rawfile_button = MyButton(self.button_frame, text="Delete", command=self.delete_selected_rawfiles,width=12, height=1)
        self.delete_rawfile_button.pack(side=tk.LEFT, padx=1, pady=5)

        # Selecting mass list file
        self.mass_list_frame = tk.Frame(self.rawfiles_frame)
        self.mass_list_frame.pack(fill=tk.BOTH, padx = 5)
        self.mass_list_label = tk.Label(self.mass_list_frame, text = 'Transition list file:')
        self.mass_list_label.pack(side=tk.TOP, anchor = tk.W)
        self.mass_list_path_entry = tk.Entry(self.mass_list_frame, textvariable=self.mass_list_path)
        self.mass_list_path_entry.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.select_mass_list_button = MyButton(self.mass_list_frame, text="Select Mass List File", command=self.select_mass_file)
        self.select_mass_list_button.pack()

        # Selecting output directory
        self.output_file_path_entry_frame = tk.Frame(self.rawfiles_frame)
        self.output_file_path_entry_frame.pack(side=tk.BOTTOM, anchor = tk.S, fill=tk.BOTH, padx = 5)
        self.output_file_path_entry_label = tk.Label(self.output_file_path_entry_frame, text = 'Output file directory:')
        self.output_file_path_entry_label.pack(side=tk.TOP, anchor = tk.W)
        self.output_file_path_entry_box = tk.Entry(self.output_file_path_entry_frame, textvariable=self.output_file_path)
        self.output_file_path_entry_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.output_file_path_entry_button = MyButton(self.output_file_path_entry_frame, text="Select Output Folder", command=self.select_output_file_path)
        self.output_file_path_entry_button.pack()

        # right frame
        self.params_frame = tk.Frame(self)
        self.params_frame.pack(side=tk.RIGHT, anchor = tk.N, fill=tk.BOTH, pady=(30,10))

        # checkboxes for identifying ms2 and mobility data
        self.checkbutton_frame = tk.Frame(self.params_frame)
        self.checkbutton_frame.pack(side=tk.TOP)

        self.is_MS2_ckbox = tk.Checkbutton(self.checkbutton_frame, text="Contains MS2 Data", variable = self.is_MS2_var, command = self.fill_param_box)
        self.is_MS2_ckbox.pack(anchor=tk.W, padx = (0, 20))
        self.is_MS2_ckbox.bind('<Return>', lambda event: self.toggle_checkbox(self.is_MS2_var, event))

        self.is_mob_ckbox = tk.Checkbutton(self.checkbutton_frame, text="Contains Ion Mobility Data", variable = self.is_mob_var, command = self.fill_param_box)
        self.is_mob_ckbox.pack(anchor=tk.W, padx = (0,20))
        self.is_mob_ckbox.bind('<Return>', lambda event: self.toggle_checkbox(self.is_mob_var, event))

        # tolerance values and image dimension parameters
        self.parameters_txt_frame = tk.Frame(self.params_frame)
        self.parameters_txt_frame.pack(side=tk.TOP, fill=tk.BOTH, pady = (10,0))
        self.initialize_param_box()
        self.fill_param_box()

        # Run button
        self.run_button_border = tk.Frame(self.params_frame, highlightbackground = "red",  
                         highlightthickness = 2, bd=0) 
        self.run_workflow_button = MyButton(self.run_button_border, text="RUN", command=self.run_workflow, \
                                             fg = 'red', font=(None, 20, 'bold'))
        self.run_button_border.pack(side = tk.BOTTOM, anchor = tk.S, fill=tk.BOTH, padx = 10)
        self.run_workflow_button.pack(side = tk.BOTTOM, anchor = tk.S, fill=tk.BOTH)
        self.run_workflow_button.config(height = 1)

        wheel = ttk.Progressbar(self, orient='horizontal')

    def destroy_all_windows(self):
        try:
            self.file_explorer.destroy()
        except:
            pass
        self.destroy()

    def toggle_checkbox(self, checkbutton, event=None):
        state = checkbutton.get()
        checkbutton.set(not state)

    def select_output_file_path(self):
        """Opens a dialog box to select directory to save files to"""
        self.output_file_path.set(filedialog.askdirectory())

    def initialize_param_box(self):
        """Sets up the box containing tolerances and image dimension inputs"""
        self.tolerance_label = tk.Label(self.parameters_txt_frame, text='Tolerance values:')
        self.tolerance_label.pack(anchor=tk.W)

        self.tolerance_textboxes = [[None,None,None] for _ in range(4)] # Each entry corresponds to specific tolerance label, value, and unit entry
        self.tolerance_parameter_labels = ["MS1 Mass", "Precursor Mass", "Fragment Mass", "Ion Mobility"]  # Labels for the entry widgets
        self.tolerance_default_values = ['10.0', '1.0', '10.0', '0.1']
        self.tolerance_default_units = ['ppm','m/z','ppm','μs']
        self.tolerance_allowed_units = [['ppm','m/z'],['ppm','m/z'],['ppm','m/z'],['μs','1/K0']]
        
        self.tolerance_value = [tk.StringVar(self) for i in range(4)]
        self.tolerance_units = [tk.StringVar(self) for i in range(4)]

        self.tol_frames = [tk.Frame(self.parameters_txt_frame) for i in range(4)]

        for i in range(len(self.tolerance_textboxes)):    
            value = self.tolerance_parameter_labels[i]
            self.tolerance_textboxes[i][0] = tk.Label(self.tol_frames[i], text=value)
            
            self.tolerance_value[i].set(self.tolerance_default_values[i])
            self.tolerance_textboxes[i][1] = tk.Entry(self.tol_frames[i], textvariable=self.tolerance_value[i], width = 15)

            self.tolerance_units[i].set(self.tolerance_default_units[i])
            self.tolerance_textboxes[i][2] = ttk.OptionMenu(self.tol_frames[i], self.tolerance_units[i], self.tolerance_default_units[i], *self.tolerance_allowed_units[i])

            # self.tolerance_textboxes[i][2] = tk.Entry(self.tol_frames[i], width = 5)
            # self.tolerance_textboxes[i][2].insert(0, value)
        
        self.img_dim_frames = tk.Frame(self.params_frame)
        self.img_dim_frames.pack(side=tk.TOP, fill=tk.BOTH, pady = (10,10))

        self.img_dim_label = tk.Label(self.img_dim_frames, text='Image dimensions (h x w):')
        self.img_h_entry_box = tk.Entry(self.img_dim_frames, textvariable=self.img_h, width = 6)
        self.unnecessary_x_label = tk.Label(self.img_dim_frames, text='x')
        self.img_w_entry_box = tk.Entry(self.img_dim_frames, textvariable=self.img_w, width = 6)
        self.img_dim_unit_label = tk.Label(self.img_dim_frames, text='mm')

        self.img_dim_label.pack(anchor=tk.W)
        self.img_h_entry_box.pack(side=tk.LEFT)
        self.unnecessary_x_label.pack(side=tk.LEFT)
        self.img_w_entry_box.pack(side=tk.LEFT)
        self.img_dim_unit_label.pack(side=tk.LEFT)

    def fill_param_box(self, event = None):
        # Remove any displayed widgets so widgets arent packed in a different order
        for index, i in enumerate(self.tolerance_textboxes):
            for j in i:
                j.pack_forget()
                self.tol_frames[index].pack_forget()

        for index, i in enumerate(self.tolerance_textboxes):
            if index in [1,2]:
                if not self.is_MS2_var.get():
                    continue
            elif index == 3:
                if not self.is_mob_var.get():
                    continue
            self.tol_frames[index].pack(anchor = tk.W)
            i[0].pack(anchor=tk.W)  # Pack label widget above
            i[1].pack(side=tk.LEFT)  # Pack entry value widget to the left
            i[2].pack(side=tk.LEFT)  # Pack entry unit widget to the left

    def select_mass_file(self):
        filetypes = [("csv or Excel", "*.txt;*.csv;*.xlsx;*.xls"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.mass_list_path.set(file_path)

    def open_file_explorer(self):
        self.open_explorer_button['state'] = 'disabled'
        self.file_explorer = FileExplorerWindow(self.receive_raw_files)
        self.file_explorer.mainloop()

    def receive_raw_files(self, raw_files):
        # print("Received raw files:", raw_files.get())
        self.rawfile_paths.set(raw_files.get())

        for i in self.rawfile_paths.get().split('|'):
            self.rawfiles_box.insert(tk.END, str(i))
        self.rawfile_paths.set('|'.join(self.rawfiles_box.get(0, tk.END)))
        self.open_explorer_button['state'] = 'normal'

    def delete_selected_rawfiles(self):
        selected_indices = self.rawfiles_box.curselection()
        for index in selected_indices[::-1]: # Iterate in reverse order to avoid index shifting
            self.rawfiles_box.delete(index)
        joined_paths = '|'.join(self.rawfiles_box.get(0, tk.END))
        if joined_paths == '':
            self.rawfiles_box.delete(0, tk.END)
        self.rawfile_paths.set(joined_paths)

    def get_input_vars(self):
        # block button
        try:
            self.run_workflow_button['state'] = 'disabled'

            example_file = self.rawfile_paths.get().split('|')
            example_file, filenames_checked = verify_rawfile_names_gui(example_file)
            assert filenames_checked

            mass_list_dir = self.mass_list_path.get()
            if not Path(mass_list_dir).exists():
                error_message = "Mass list file given does not exist."
                tk.messagebox.showerror("Select file error", error_message)
                self.run_workflow_button['state'] = 'normal'

            assert Path(mass_list_dir).exists()

            try:
                mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance = [float(i.get()) for i in self.tolerance_value]
            except:
                error_message = "Tolerance values must be numbers."
                tk.messagebox.showerror("Argument error", error_message)
                self.run_workflow_button['state'] = 'normal'

            mass_tolerance_MS1_units, mass_tolerance_prec_units, mass_tolerance_frag_units, mobility_tolerance_units = [i.get() for i in self.tolerance_units]

            try:
                img_height, img_width = float(self.img_h.get()), float(self.img_w.get())
                image_dimensions_units = "mm"
            except: 
                error_message = "Image dimension values must be numbers."
                tk.messagebox.showerror("Argument error", error_message)
                self.run_workflow_button['state'] = 'normal'


            is_MS2, is_mobility = bool(self.is_MS2_var.get()), bool(self.is_mob_var.get())

            normalize_img_sizes = True

            output_file_loc = self.output_file_path.get()
            if not output_file_loc:
                output_file_loc = None

            return example_file, mass_list_dir, mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance,\
                mass_tolerance_MS1_units, mass_tolerance_prec_units, mass_tolerance_frag_units, mobility_tolerance_units,\
                img_height, img_width, image_dimensions_units, is_MS2, is_mobility, normalize_img_sizes, output_file_loc

        except:
            self.run_workflow_button['state'] = 'normal'


    def run_workflow(self):
        try:
            self.file_explorer.destroy()
            self.open_explorer_button['state'] = 'normal'
        except:
            pass
        
        try:
            example_file, mass_list_dir, mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance,\
                mass_tolerance_MS1_units, mass_tolerance_prec_units, mass_tolerance_frag_units, mobility_tolerance_units,\
                img_height, img_width, image_dimensions_units, is_MS2, is_mobility, normalize_img_sizes, output_file_loc,\
                    = self.get_input_vars()
            print(mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance)
            print(mass_tolerance_MS1_units, mass_tolerance_prec_units, mass_tolerance_frag_units, mobility_tolerance_units)
            self.metadata = msigen.get_metadata_and_params(example_file, mass_list_dir, mass_tolerance_MS1, mass_tolerance_MS1_units, mass_tolerance_prec, \
                        mass_tolerance_prec_units, mass_tolerance_frag, mass_tolerance_frag_units, mobility_tolerance, mobility_tolerance_units,\
                        img_height, img_width, image_dimensions_units, is_MS2, is_mobility, normalize_img_sizes, output_file_loc, in_jupyter = False, testing = True)

            self.open_progessbar_window()

            self.results = {}
            tkinter_widgets = [self.prog_bar, self.current_operation_label, self.prog_label]

            self.MSIGen_process = Thread(target = msigen.get_image_data, args = (self.metadata, False), \
                kwargs = {'in_jupyter':False, 
                        'testing':False, \
                        'gui':True, \
                        'results':self.results, \
                        'tkinter_widgets':tkinter_widgets})
            self.MSIGen_process.start()
        
            self.monitor_progressbar()

        except Exception as error:
                tk.messagebox.showerror("Error", error)

        finally:            
            # return button to normal state
            self.run_workflow_button['state'] = 'normal'

    def open_progessbar_window(self):
        self.progress_bar_window = tk.Toplevel(self)
        self.progress_bar_window.attributes('-topmost', True)
        self.progress_bar_window.geometry("300x200")
        self.progress_bar_window.title("Extracting data...")
        self.prog_bar = ttk.Progressbar(self.progress_bar_window, length = 250, orient='horizontal')
        self.prog_bar.pack()
        self.current_operation_label = tk.Label(self.progress_bar_window)
        self.current_operation_label.pack()
        self.prog_label = tk.Label(self.progress_bar_window)
        self.prog_label.pack()
        self.start_time = time()

    def monitor_progressbar(self):
        try:
            if self.MSIGen_process.is_alive():
                # check again after 200ms (0.2s)
                self.after(200, self.monitor_progressbar)
            else:
                t_tot = time()-self.start_time
                t_min = t_tot//60
                t_s = round(t_tot - (t_min*60), 2)

                if t_min:
                    self.prog_label['text'] = f'Time elapsed: {t_min} min {t_s} s'
                else:
                    self.prog_label['text'] = f'Time elapsed: {t_s} s'

                self.metadata, self.pixels = self.results['metadata'], self.results['pixels']

                self.current_operation_label['text'] = "Complete!"
                self.progress_bar_window.protocol("WM_DELETE_WINDOW", self.open_image_maker)
                self.continue_to_visualization_button = MyButton(self.progress_bar_window, text="Finish", command=self.open_image_maker, width=12, height=1)
                self.continue_to_visualization_button.pack()
        
        except Exception as error:
            tk.messagebox.showerror("Error", error)
            self.run_workflow_button['state'] = 'normal'
            self.progress_bar_window.destroy()

    def open_image_maker(self):
        """Opens the window that contains all parameters needed to export images.
It includes 3 tabs:
    1: For creating ion images
    2: For creating fractional images
    3: For creating ratio images

Images can be saved as figures (containing a title and colorbar), images, or arrays and can be saved using a selection of colormaps.
The brightness of the image can be scaled by a percentile or an absolute threshold.
The mass list can be viewed to obtain mass list entry indices."""
        self.progress_bar_window.destroy()
        self.withdraw()
        self.image_maker_window = tk.Toplevel(self)
        self.image_maker_window.protocol("WM_DELETE_WINDOW", self.reselect_raw_files)
        self.notebook = ttk.Notebook(self.image_maker_window)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text='Ion Images')
        self.notebook.add(self.tab2, text='Fractional Images')
        self.notebook.add(self.tab3, text='Ratio Images')

        # Tab 1: Ion images
        self.normalization1_label = tk.Label(self.tab1, text = "Normalization method:")
        self.normalization1_label.grid(row=0, column=0, sticky = "e")
        self.dropdown_normalization1_var = tk.StringVar(value="None")
        self.dropdown_normalization1 = ttk.OptionMenu(self.tab1, self.dropdown_normalization1_var, \
            "None", *["None", "TIC", "Internal Standard"], command=self.show_or_hide_std_idx_entry)
        self.dropdown_normalization1.grid(row=0, column=1, sticky = "w")

        self.std_idx_var_label = tk.Label(self.tab1, text = "Index of internal standard from mass list:")
        self.std_idx_var_label.grid(row=1, column=0, sticky = "e")
        self.std_idx_var = tk.StringVar(value="1")
        self.std_idx_entry = tk.Entry(self.tab1, textvariable=self.std_idx_var)
        if self.dropdown_normalization1_var.get() == "Internal Standard":
            self.std_idx_entry.grid(row=1, column=1, sticky = "ew")

        self.choose_scale_threshold_label1 = tk.Label(self.tab1, text = "Reduce max intensity to a percentile or an absolute value?")
        self.choose_scale_threshold_label1.grid(row=2, column=0, sticky = "e")
        self.choose_scale_threshold_var1 = tk.StringVar(value="Percentile")
        self.choose_scale_threshold_dropdown1 = ttk.OptionMenu(self.tab1, self.choose_scale_threshold_var1, "Percentile", *["Percentile", "Absolute"],\
            command = lambda selection: self.scale_or_threshold_display(selection, self.scale_label1, self.scale_entry1, self.threshold_label1, self.threshold_entry1, 3))
        self.choose_scale_threshold_dropdown1.grid(row=2, column=1, sticky = "w")

        self.scale_label1 = tk.Label(self.tab1, text = "Adjust max intensity to this percentile:")
        self.scale_stringvar = tk.StringVar(value=str(self.scale.get()))
        self.scale_entry1 = tk.Entry(self.tab1, textvariable=self.scale_stringvar)
        self.scale_label1.grid(row=3, column=0, sticky = "e")
        self.scale_entry1.grid(row=3, column=1, sticky = "ew")

        self.threshold_label1 = tk.Label(self.tab1, text = "Adjust max intensity to this value:")
        self.threshold_stringvar = tk.StringVar(value="1")
        self.threshold_entry1 = tk.Entry(self.tab1, textvariable=self.threshold_stringvar)

        self.tab1.columnconfigure(0, weight=1, uniform = 'half')
        self.tab1.columnconfigure(1, weight=1, uniform = 'half')

        # Tab 2: Fractional Images
        self.normalization2_label = tk.Label(self.tab2, text = "Normalization method:")
        self.normalization2_label.grid(row=0, column=0, sticky = "e")
        self.dropdown_normalization2_var = tk.StringVar(value="None")
        self.dropdown_normalization2 = ttk.OptionMenu(self.tab2, self.dropdown_normalization2_var, "None", *["None", "Base Peak"])
        self.dropdown_normalization2.grid(row=0, column=1, sticky = "w")

        self.frac_img_idxs_label = tk.Label(self.tab2, text = "Indices of ions to use from mass list:")
        self.frac_img_idxs_label.grid(row=1, column=0, sticky = "e")
        self.frac_img_idxs_var = tk.StringVar(value="1, 2")
        self.frac_img_idxs = tk.Entry(self.tab2, textvariable=self.frac_img_idxs_var)
        self.frac_img_idxs.grid(row=1, column=1, sticky = "ew")

        self.choose_scale_threshold_label2 = tk.Label(self.tab2, text = "Reduce max intensity to a percentile or an absolute value?")
        self.choose_scale_threshold_label2.grid(row=2, column=0, sticky = "e")
        self.choose_scale_threshold_var2 = tk.StringVar(value="Percentile")
        self.choose_scale_threshold_dropdown2 = ttk.OptionMenu(self.tab2, self.choose_scale_threshold_var2, "Percentile", *["Percentile", "Absolute"],\
            command = lambda selection: self.scale_or_threshold_display(selection, self.scale_label2, self.scale_entry2, self.threshold_label2, self.threshold_entry2, 3))
        self.choose_scale_threshold_dropdown2.grid(row=2, column=1, sticky = "w")

        self.scale_label2 = tk.Label(self.tab2, text = "Adjust max intensity to this quantile (0-1):")
        self.scale_label2.grid(row=3, column=0, sticky = "e")
        self.scale_entry2 = tk.Entry(self.tab2, textvariable=self.scale_stringvar)
        self.scale_entry2.grid(row=3, column=1, sticky = "ew")

        self.threshold_label2 = tk.Label(self.tab2, text = "Adjust max intensity to this value:")
        self.threshold_entry2 = tk.Entry(self.tab2, textvariable=self.threshold_stringvar)

        self.tab2.columnconfigure(0, weight=1, uniform = 'half')
        self.tab2.columnconfigure(1, weight=1, uniform = 'half')

        # Tab 3: Ratio Images
        self.normalization3_label = tk.Label(self.tab3, text = "Normalization method:")
        self.normalization3_label.grid(row=0, column=0, sticky = "e")
        self.dropdown_normalization3_var = tk.StringVar(value="None")
        self.dropdown_normalization3 = ttk.OptionMenu(self.tab3, self.dropdown_normalization3_var, "None", *["None", "Base Peak"])
        self.dropdown_normalization3.grid(row=0, column=1, sticky = "w")

        self.ratio_img_idxs_label = tk.Label(self.tab3, text = "Indices of ions to use from mass list:")
        self.ratio_img_idxs_label.grid(row=1, column=0, sticky = "e")
        self.ratio_img_idxs_var = tk.StringVar(value="1, 2")
        self.ratio_img_idxs = tk.Entry(self.tab3, textvariable=self.ratio_img_idxs_var)
        self.ratio_img_idxs.grid(row=1, column=1, sticky = "ew")

        self.choose_scale_threshold_label3 = tk.Label(self.tab3, text = "Reduce max intensity to a percentile or an absolute value?")
        self.choose_scale_threshold_label3.grid(row=2, column=0, sticky = "e")
        self.choose_scale_threshold_var3 = tk.StringVar(value="Percentile")
        self.choose_scale_threshold_dropdown3 = ttk.OptionMenu(self.tab3, self.choose_scale_threshold_var3, "Percentile", *["Percentile", "Absolute"],\
            command = lambda selection: self.scale_or_threshold_display(selection, self.scale_label3, self.scale_entry3, self.threshold_label3, self.threshold_entry3, 3))
        self.choose_scale_threshold_dropdown3.grid(row=2, column=1, sticky = "w")

        self.scale_label3 = tk.Label(self.tab3, text = "Adjust max intensity to this quantile (0-1):")
        self.scale_label3.grid(row=3, column=0, sticky = "e")
        self.scale_entry3 = tk.Entry(self.tab3, textvariable=self.scale_stringvar)
        self.scale_entry3.grid(row=3, column=1, sticky = "ew")

        self.threshold_label3 = tk.Label(self.tab3, text = "Adjust max intensity to this value:")
        self.threshold_entry3 = tk.Entry(self.tab3, textvariable=self.threshold_stringvar)

        self.handle_infinity_method_label = tk.Label(self.tab3, text = "How to handle divide by zero errors:")
        self.handle_infinity_method_label.grid(row=4, column=0, sticky = "e")
        self.handle_infinity_method_var = tk.StringVar(value="Maximum")
        self.dropdown_handle_infinity_method = ttk.OptionMenu(self.tab3, self.handle_infinity_method_var, "Maximum", *['Maximum', 'Infinity', 'Zero'])
        self.dropdown_handle_infinity_method.grid(row=4, column=1, sticky = "w")

        self.log_scale_var = tk.IntVar()
        self.log_scale_ckbtn = tk.Checkbutton(self.tab3, text="Use log-scale for intensity", variable = self.log_scale_var)
        self.log_scale_ckbtn.grid(row = 5, column = 0, columnspan = 2)

        self.tab3.columnconfigure(0, weight=1, uniform = 'half')
        self.tab3.columnconfigure(1, weight=1, uniform = 'half')

        # Parameters applying to any image type
        self.general_img_params_frame = tk.Frame(self.image_maker_window)
        self.general_img_params_frame.pack(fill="x", expand=True, padx = 10)
        self.dropdown_colormap_label = tk.Label(self.general_img_params_frame, text = "Colormap to use:")
        self.dropdown_colormap_label.grid(row=0, column=0, sticky = "e", padx = 5)
        self.dropdown_colormap_var = tk.StringVar(value="viridis")
        self.dropdown_colormap = ttk.OptionMenu(self.general_img_params_frame, self.dropdown_colormap_var, "viridis", *["viridis", "cividis", "hot", "jet", "seismic"])
        self.dropdown_colormap.grid(row=0, column=1, sticky = "w", padx = 5)

        self.dropdown_savetype_label = tk.Label(self.general_img_params_frame, text = "Save images as:")
        self.dropdown_savetype_label.grid(row=1, column=0, sticky = "e", padx = 5)
        self.dropdown_savetype_var = tk.StringVar(value="figure")
        self.dropdown_savetype = ttk.OptionMenu(self.general_img_params_frame, self.dropdown_savetype_var, "figure", *["figure", "image", "array"])
        self.dropdown_savetype.grid(row=1, column=1, sticky = "w", padx = 5)

        self.output_file_path_label2 = tk.Label(self.general_img_params_frame, text = "Path to save images to:")
        self.output_file_path_label2.grid(row=2, column=0, sticky = "w", padx = 5)
        self.output_file_path_entry_box2 = tk.Entry(self.general_img_params_frame, textvariable=self.output_file_path)
        self.output_file_path_entry_box2.grid(row=2, column=1, sticky = "ew", padx = 5)
        self.output_file_path_entry_button2 = MyButton(self.general_img_params_frame, text="Reselect Output Folder", command=self.select_output_file_path)
        self.output_file_path_entry_button2.grid(row=3, column=1, sticky = "w", padx = 5)

        self.general_img_params_frame.columnconfigure(1, weight=1, uniform = 'half')

        self.img_maker_buttons_frame = tk.Frame(self.image_maker_window)
        self.img_maker_buttons_frame.pack()
        
        self.generate_images_button = MyButton(self.img_maker_buttons_frame, text = 'Generate Images', command = self.generate_images)
        self.generate_images_button.pack(side = tk.LEFT, padx = 5)

        self.view_mass_list_button = MyButton(self.img_maker_buttons_frame, text = 'View Mass List', command = self.display_mass_list)
        self.view_mass_list_button.pack(side = tk.LEFT, padx = 5)

        self.reselect_raw_files_button = MyButton(self.img_maker_buttons_frame, text = 'Reselect Raw Files', command = self.reselect_raw_files)
        self.reselect_raw_files_button.pack(side = tk.LEFT, padx = 5)

        # self.end_all_button = MyButton(self.img_maker_buttons_frame, text = 'Finish')
        # self.end_all_button.pack(side = tk.LEFT)

    def generate_images(self):
        """Exports images based on the active tab and inputted parameters"""
        active_nb_pg = self.notebook.tab(self.notebook.select(),"text")
        
        if active_nb_pg == "Ion Images":
            scale, threshold = self.get_scale_threshold_values(self.choose_scale_threshold_var1, \
                                                        self.scale_stringvar, self.threshold_stringvar)
            print(scale, threshold)

            std_idx = int(self.std_idx_var.get())
            print(f'std_idx: {std_idx}')

            if self.dropdown_normalization1_var.get() == 'Internal Standard':
                try:
                    assert std_idx > 0
                except:
                    std_idx = 0
                    error_message = "The index of the internal standard must be a single positive integer."
                    tk.messagebox.showerror("Internal standard index error", error_message)
            else:
                std_idx = 1

            if std_idx:
                pixels_normed = vis.get_pixels_to_display(self.pixels, self.metadata, normalize = self.dropdown_normalization1_var.get(), std_idx = std_idx)
                vis.display_images(pixels_normed, self.metadata, MSI_data_output=self.output_file_path.get(), cmap=self.dropdown_colormap_var.get(),\
                    threshold=threshold, scale=scale, save_imgs=True, image_savetype=self.dropdown_savetype_var.get())
                self.open_images_were_saved_dialog()

        elif active_nb_pg == "Fractional Images":
            scale, threshold = self.get_scale_threshold_values(self.choose_scale_threshold_var2, \
                                        self.scale_stringvar, self.threshold_stringvar)
            try:
                idxs_list = [int(i) for i in self.frac_img_idxs_var.get().split(',')]
                assert all([i>=0 for i in idxs_list])
            except:
                idxs_list = []
                error_message = "The indices given must be positive integers separated by a ','"
                tk.messagebox.showerror("Index error", error_message)

            if idxs_list:
                fract_imgs = vis.get_fractional_abundance_imgs(self.pixels, self.metadata, idxs = idxs_list, \
                    normalize = self.dropdown_normalization2_var.get())
                vis.display_fractional_images(fract_imgs, self.metadata, save_imgs = True, MSI_data_output = self.output_file_path.get(), \
                    cmap = self.dropdown_colormap_var.get(), image_savetype=self.dropdown_savetype_var.get(), scale=scale, threshold=threshold)
                self.open_images_were_saved_dialog()

        elif active_nb_pg == "Ratio Images":
            scale, threshold = self.get_scale_threshold_values(self.choose_scale_threshold_var3, \
                                        self.scale_stringvar, self.threshold_stringvar)
            try:
                idxs_list = [int(i) for i in self.ratio_img_idxs_var.get().split(',')]
                assert all([i>0 for i in idxs_list]) and (len(idxs_list) == 2)
            except:
                idxs_list = []
                error_message = "The indices given must be two positive integers separated by a ','."
                tk.messagebox.showerror("Index error", error_message)

            if idxs_list:
                ratio_imgs = vis.get_ratio_imgs(self.pixels, self.metadata, idxs = idxs_list, \
                    normalize = self.dropdown_normalization3_var.get(), handle_infinity = self.handle_infinity_method_var.get())
                vis.display_ratio_images(ratio_imgs, self.metadata, save_imgs = True, \
                    MSI_data_output = self.output_file_path.get(), cmap = self.dropdown_colormap_var.get(),\
                    log_scale = bool(self.log_scale_var.get()), scale=scale, threshold=threshold, \
                    image_savetype=self.dropdown_savetype_var.get())
                self.open_images_were_saved_dialog()

    def open_images_were_saved_dialog(self):
        """A window that contains a hyperlink to the folder the images were exported to."""
        self.images_were_saved_dialog = tk.Toplevel(self.image_maker_window)
        self.images_were_saved_dialog.minsize(200,100)
        self.images_were_saved_dialog.protocol("WM_DELETE_WINDOW", self.images_were_saved_dialog.destroy)

        self.saved_imgs_label1 = tk.Label(self.images_were_saved_dialog, text="Your files were saved to:")
        self.saved_imgs_label1.pack(side = tk.TOP)
        self.saved_imgs_label2 = tk.Label(self.images_were_saved_dialog, text=self.output_file_path.get()+'/images', fg="blue", cursor="hand2")
        self.saved_imgs_label2.bind("<Button-1>", self.open_hyperlink)
        self.saved_imgs_label2.pack(side = tk.TOP)

    def open_hyperlink(self, *args):
        os.startfile(self.output_file_path.get()+'/images')

    def display_mass_list(self):
        columns, output_table = get_final_mass_list_gui(self.metadata)
        self.mass_list_window = tk.Toplevel(self)
        self.mass_list_tree = ttk.Treeview(self.mass_list_window, columns = columns, show="headings")
        self.mass_list_tree.pack(fill="both", expand=True)
        for heading in columns:
            self.mass_list_tree.heading(heading, text = heading)
        for row in output_table:
            self.mass_list_tree.insert("","end", values=tuple(row))

    def reselect_raw_files(self):
        """Goes back to file selection screen. All progress will be lost."""
        self.image_maker_window.destroy()
        self.deiconify()

    def show_or_hide_std_idx_entry(self, *args):
        """Hides the std_idx entrybox when intl_std normalization is not selected"""
        if self.dropdown_normalization1_var.get() == "Internal Standard":
            self.std_idx_var_label.grid(row=1, column=0, sticky = "e")
            self.std_idx_entry.grid(row=1, column=1, sticky = "ew")
        else: 
            self.std_idx_var_label.grid_forget()
            self.std_idx_entry.grid_forget()

    def scale_or_threshold_display(self, selection, scale_label, scale_entry, threshold_label, threshold_entry, row):
        """Toggles the display between percentile and threshold depending on currently selected dropdown value"""
        if selection == "Percentile":
            scale_label.grid(row=row, column=0, sticky = "e")
            scale_entry.grid(row=row, column=1, sticky = "ew")
            threshold_label.grid_forget()
            threshold_entry.grid_forget()
        else: 
            threshold_label.grid(row=row, column=0, sticky = "e")
            threshold_entry.grid(row=row, column=1, sticky = "ew")
            scale_label.grid_forget()
            scale_entry.grid_forget()

    def get_scale_threshold_values(self, dropdown_menu_var, scale_stringvar, threshold_stringvar):
        """Gets the appropriate threshold or percentile to scale the image intensity to for later use."""
        if dropdown_menu_var.get() == "Percentile":
            scale = scale_stringvar.get()
            threshold = None
            try:
                scale = float(scale)/100
            except:
                scale = 1
        else:
            scale = 1
            threshold = threshold_stringvar.get()
            try:
                threshold = float(threshold)
            except:
                threshold = None
        return scale, threshold

# TODO make this easier to use.
class FileExplorerWindow(tk.Tk):
    """File explorer that allows .d data to be treated as files rather than folders."""
    def __init__(self, callback):
        super().__init__()
        self.title("Insert Listbox Example")
        self.geometry("400x300")  # Set a fixed size for the window
        self.callback = callback
        self.selected_items = tk.StringVar(value="")
        self.selected_drive = tk.StringVar(value="")
        self.raw_files = tk.StringVar(value="")
        self.current_directory = Path.cwd()
        ## TODO implement a good redo and undo for file navigation
        # self.movement_history = tk.StringVar(value="")
        # self.movement_future = tk.StringVar(value="")

        self.drives = win32api.GetLogicalDriveStrings()
        self.drives = self.drives.split('\000')[:-1]
        self.protocol("WM_DELETE_WINDOW", self.close_raw_file_selection_window)

        self.add_listboxes()

    def get_current_directory_contents(self):
        """Gets the current files and folders in the selected directory for display"""
        self.listbox.delete(0, tk.END)
        self.dir_contents = []
        self.textbox_contents.set(str(self.current_directory))

        if self.dropdown_var.get() == "all files":
            for i in self.current_directory.glob('*'):
                if not i.name.startswith(('__','~','.','$')):    
                    self.dir_contents.append(i.name)

        else: 
            for i in self.current_directory.glob('*'):
                if not i.name.startswith(('__','~','.','$')):
                    if i.is_dir():
                        self.dir_contents.append(i.name)
                    else:
                        if i.name.lower().endswith(('.d','.mzml','.raw')): 
                            self.dir_contents.append(i.name)

        for i in self.dir_contents:
            self.listbox.insert(tk.END, str(i))        

    def add_listboxes(self):
        """Makes a box on the left side of the window that contains the commonly used directories such as:
    Drive letters, Downloads, Desktop, etc.
for easier navigation"""
        self.listbox_frame = tk.Frame(self)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.listbox_driveletters = tk.Listbox(self.listbox_frame, selectmode=tk.EXTENDED, width=20)  # Set width for the left listbox
        self.listbox_driveletters.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fill_listbox_driveletters()
        
        self.listbox_driveletters.bind("<Double-Button-1>", self.on_double_click_drives)
        # self.listbox_driveletters.bind("<ButtonRelease-1>", self.get_selected_drive_values)
        # self.listbox_driveletters.bind("<Return>", self.on_return)


        self.listbox = tk.Listbox(self.listbox_frame, selectmode=tk.EXTENDED)
        self.listbox.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.listbox.bind("<Double-Button-1>", self.on_double_click)
        self.listbox.bind("<ButtonRelease-1>", self.get_selected_values)
        self.listbox.bind("<KeyPress-BackSpace>", self.move_to_parent_dir)
        self.listbox.bind("<Return>", self.on_return)
        
        self.textbox_frame = tk.Frame(self.listbox_frame)
        self.textbox_frame.pack(side=tk.TOP, fill=tk.X, padx = 1)

        self.back_arrow_button = MyButton(self.textbox_frame, text="\u2190", command=self.move_to_parent_dir)
        self.back_arrow_button.pack(side = tk.RIGHT)

        self.textbox_contents = tk.StringVar(self.listbox_frame, value=self.current_directory)
        self.textbox = tk.Entry(self.textbox_frame, textvariable=self.textbox_contents)
        self.textbox.pack(side=tk.LEFT, fill=tk.X)
        self.textbox.bind("<Configure>", self.on_textbox_resize)
        self.textbox.bind("<Return>", self.on_textbox_return)

        # Adding the button
        self.select_button = MyButton(self, text="Select files", command=self.on_return)
        self.select_button.pack(side=tk.RIGHT, padx=10, pady=5)

        # Adding the dropdown menu
        self.dropdown_var = tk.StringVar(self)
        self.dropdown_var.set(".d, .mzML, or .raw files")  # Set default value
        self.dropdown_var.trace_add("write", self.on_dropdown_change)
        self.dropdown_menu = ttk.OptionMenu(self, self.dropdown_var, ".d, .mzML, or .raw files", *[".d, .mzML, or .raw files", "all files"])
        self.dropdown_menu.pack(side=tk.RIGHT, padx=10, pady=5)

        self.get_current_directory_contents()

    def on_dropdown_change(self, *args):
        """Allows the user to not display unselectable files."""
        selected_option = self.dropdown_var.get()
        self.get_current_directory_contents()

    def on_textbox_resize(self, event):
        """Adjust the Text widget width dynamically with Listbox. 
Allows for proper resizing of the box displaying the currently selected directory"""
        listbox_width = self.listbox.winfo_width()
        self.textbox.config(width=listbox_width)

    def on_textbox_return(self, event):
        """Goes to directory typed into the textbox or selects file if it is a file."""
        text = self.textbox_contents.get() # Retrieve text from the textbox
        text = text.replace('"','')
        if Path(text).is_dir():
            if Path(text).name.lower().endswith('.d'):
                self.raw_files.set("|".join(text.split('|')))
                self.close_raw_file_selection_window()
            else:
                self.current_directory = Path(text)
                self.get_current_directory_contents()
        
        elif Path(text).name.lower().endswith(('.d','.raw','.mzml')):
            self.raw_files.set("|".join(text.split('|')))
            self.close_raw_file_selection_window()

        # print("Textbox content:", text)

    def on_return(self, event=None):
        """Allows for navigation with Return instead of the mouse"""
        self.get_selected_values(event)

        # check if all files are 
        use_as_files = True
        for i in self.selected_items.get().split('|'):
            i = Path(self.current_directory, i)
            if not i.name.lower().endswith(('.d','.mzml','.raw',)):
                use_as_files = False
            
        if use_as_files:
            # get list of raw files as a '|' dileniated string
            self.raw_files.set("|".join([str(Path(self.current_directory, i)) for i in self.selected_items.get().split('|')]))
            self.close_raw_file_selection_window()

        elif Path(self.current_directory, self.selected_items.get().split('|')[0]).is_dir():
            self.current_directory = Path(self.current_directory, self.selected_items.get().split('|')[0])

            # clear then update contents            
            self.get_current_directory_contents()
            self.get_selected_values(event)


    def on_double_click(self, event):
        """Opens folder or selects files."""
        self.get_selected_values(event)
        if len(self.selected_items.get().split('|')) == 1:
            
            # check file name extension
            if self.selected_items.get().lower().endswith(('.d','.mzml','.raw',)):
                self.raw_files.set(str(Path(self.current_directory, self.selected_items.get())))
                self.close_raw_file_selection_window()

            elif Path(self.current_directory, self.selected_items.get()).is_dir():
                self.current_directory = Path(self.current_directory, self.selected_items.get())

                # clear then update contents
                self.get_current_directory_contents()
                self.get_selected_values(event)

    def get_selected_values(self, event=None):
        selected_indices = self.listbox.curselection()
        selected_values = [self.listbox.get(index) for index in selected_indices]
        self.selected_items.set("|".join(selected_values))

    def move_to_parent_dir(self, event=None):
        self.current_directory = self.current_directory.parent

        # clear then update contents
        self.listbox.delete(0, tk.END) 
        self.get_current_directory_contents()

        self.get_selected_values(event)
    
    # =============================================
    # for the listbox containing drive letters
    # =============================================
    
    def fill_listbox_driveletters(self):
        self.listbox_driveletters.delete(0, tk.END) 

        self.listbox_driveletters.insert(tk.END, Path.home().name)     
        self.listbox_driveletters.insert(tk.END, 'User Desktop')     
        self.listbox_driveletters.insert(tk.END, 'Public Desktop')     
        self.listbox_driveletters.insert(tk.END, 'Downloads')
        self.listbox_driveletters.insert(tk.END, 'Documents')
        self.dir_shortcuts = {'User': Path.home(),
                'User Desktop': Path.home() / 'Desktop',
                'Public Desktop': Path.home().parent/'Public/Desktop',
                'Downloads': get_download_path(),
                'Documents': Path.home() / 'Documents',
                }

        for i in self.drives:
            drive_name = f'({i})'
            self.listbox_driveletters.insert(tk.END, drive_name)
            self.dir_shortcuts[drive_name] = i
           
    def get_selected_drive_values(self, event):
        selected_indices = self.listbox_driveletters.curselection()
        selected_drive = [self.listbox_driveletters.get(index) for index in selected_indices]
        self.selected_drive.set("|".join(selected_drive))

    def on_double_click_drives(self, event):
        """Opens a drive when clicked"""
        self.get_selected_drive_values(event)
        drive = self.selected_drive.get()
        if len(drive.split('|')) == 1:

            if drive in self.dir_shortcuts.keys():
                self.selected_drive.set(self.dir_shortcuts[drive])

            self.current_directory = Path(self.selected_drive.get())
            self.get_current_directory_contents()
            self.get_selected_values(event)

    def close_raw_file_selection_window(self):
        self.callback(self.raw_files)  # Pass the raw_files string to the callback function
        self.destroy()

def run_GUI():
    """Runs the MSIGen GUI"""
    app = MasterWindow()
    app.mainloop()    

# Runs the GUI if this file is run
if __name__ == "__main__":
    app = MasterWindow()
    app.lift()
    app.attributes('-topmost',True)
    app.after_idle(app.attributes,'-topmost',False)
    app.mainloop()

