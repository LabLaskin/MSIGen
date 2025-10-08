"""This module provides a base class for MSIGen which can be subclassed to handle different file formats."""

# =================================
# imports
# =================================

# General packages
import os, sys, re
import numpy as np
import pandas as pd
from scipy.interpolate import interpn#, NearestNDInterpolator
from datetime import datetime
import json
from copy import deepcopy
from time import time
from operator import itemgetter
from skimage.transform import resize
import warnings

# GUI support
import tkinter as tk

try:
    from numba import njit
    numba_present = True
except:
    numba_present = False

try:
    jupyter_prints = True
    from IPython.display import display as iPydisplay
    from IPython.display import HTML
except:
    jupyter_prints = False

# TODO Make the code able to select based on the polarity, collision energy, etc.

# non-class specific functions:
def _display_df(df):
    """Display a DataFrame in a Jupyter notebook with a maximum height."""
    pd.set_option("display.max_rows", None)
    iPydisplay(HTML(("<div style='max-height: 400px'>" + df.to_html() + "</div>")))
    pd.set_option("display.max_rows", 30)

def custom_warning(msg, err=None, category=UserWarning, stacklevel=2):
    """
    Custom warning function that formats warnings without showing the source line.
    
    Includes filename:line + category + message,
    but omits the repeated source line.
    """
    frame = sys._getframe(stacklevel)
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno

    # Build error message if provided
    if err is not None:
        msg = f"{msg}\n{type(err).__name__}: {err}"

    # Format like standard warnings
    formatted = warnings.formatwarning(msg, category, filename, lineno, line=None)
    # Remove trailing repeated source line
    formatted = formatted.splitlines()[0] + "\n"

    sys.stderr.write(formatted)
    sys.stderr.flush()

# Used to suppress useless and repeated warnings
class HiddenPrints:
    """Allows code to be run without displaying messages."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Used to save numpy data to Json
class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class get_confirmation():
    """Dialog to confirm overwriting existing files."""
    def __init__(self, file_list):
        self.master = tk.Tk()
        self.master.wm_attributes("-topmost", True)
        self.master.focus_force()
        # self.master.geometry("300x200") 
        self.response = "N"

        self.text = "The following files already exist:\n\n{}\n\nDo you want to continue and overwrite them?".format('\n'.join(file_list))

        self.label = tk.Label(self.master, text=self.text).pack()
        self.yes_button = tk.Button(self.master, text="Yes", command= lambda: self.callback("Y")).pack(side="left", anchor='e', expand = True)
        self.no_button = tk.Button(self.master, text="No", command= lambda: self.callback("N")).pack(side="right", anchor='w', expand = True)

        self.master.mainloop()
        
    def callback(self, value):
        """Get the user entry and exit the overwrite confirmation window."""
        self.response=value
        self.master.destroy()

class MSIGen_base(object):
    """
    Base class for MSIGen. This class is not generally to be used directly.

    It is intended to be subclassed for specific file formats (e.g., D, mzml, raw).

    Args:
        example_file (str, list, or tuple): 
            The file path or list of file paths to be processed. (default None)
            If type is str, it should be a single file path and all other files with 
            the same name apart from the line number will be used. 
            If the type is list or tuple, the provided files will be the only ones processed.
            None initializes the base class without data files.
        mass_list_dir (str): 
            The directory containing the mass list file. (default None)
        tol_MS1 (float): 
            Tolerance for MS1 mass selection. (default 10.)
        tol_MS1_u (str): 
            Units for MS1 tolerance ('ppm' or 'mz'). (default 'ppm')
        tol_prec (float): 
            Tolerance for precursor mass selection in MS2 entries of the mass list. (default 1.)
        tol_prec_u (str): 
            Units for precursor tolerance ('ppm' or 'mz'). (default 'mz')
        tol_frag (float): 
            Tolerance for fragment mass selection in MS2 entries of the mass list. (default 10.)
        tol_frag_u (str): 
            Units for fragment tolerance ('ppm' or 'mz'). (default 'ppm')
        tol_mob (float): 
            Tolerance for mobility selection. Ignored if no mobility data is present. (default 0.1)
        tol_mob_u (str): 
            Units for mobility tolerance ('μs' or '1/K0'). (default 'μs')
        h (float): 
            Height of the image in specified units. (default 10.)
        w (float):
            Width of the image in specified units. (default 10.)
        hw_units (str):
            Units for height and width. (default "mm") 
        is_MS2 (bool): 
            Flag indicating if the data files contain MS2 information. (default False)
        is_mobility (bool): 
            Flag indicating if the data files contain mobility information. (default False)
        normalize_img_sizes (bool): 
            Flag indicating if image sizes should be normalized. (default True)
            If True, all images will be resized to the same size and the data can be saved as an .npy or .csv file.
            If False, images will be saved in their original sizes and the data will be saved as an .npz file. 
        pixels_per_line (str):  
            Number of pixels per line ('mean', 'min', 'max', or a specific integer). (default "mean")
            If "mean", the mean number of pixels per line will be used.
            If "min", the minimum number of pixels per line will be used.
            If "max", the maximum number of pixels per line will be used.
            If an integer, that number of pixels will be used.
        output_file_loc (str): 
            Location to save the output files. (default None)
        in_jupyter (bool): 
            Flag indicating if the code is running in a Jupyter notebook. (default True)
        testing (bool): 
            Flag indicating if the code is in testing mode. (default False)
        gui (bool): 
            Flag indicating if the GUI is being used. (default False)
        save_file_format (str):
            Format to save the output files ('npy', 'npz', or 'csv'). (default "npy")
            If "npy", the data will be saved as a .npy file, or as an npz file if normalize_img_sizes is False and MS2 is True.
            If "npz", the data will be saved as a .npz file.
            If "csv", the data will be saved as a .csv file. normalize_img_sizes will be set to True in this case.
        ask_confirmation (bool):
            Flag indicating if the user should be asked for confirmation before overwriting existing files. (default True)

    Other Attributes:
        self.results (dict): 
            Dictionary to store results and metadata for the GUI. (default {})
        self.HiddenPrints (HiddenPrints):
            A context manager to suppress print statements.
        self.data_format (str, None): 
            The data format of the .d files to be read. (default None) 
            Unnecessary unless using .d files. 
        self.NpEncoder (NpEncoder):
            A custom JSON encoder for numpy data types.
        self.get_confirmation_dialogue (get_confirmation):
            A dialog to confirm overwriting existing files.
        self.numba_present (bool):
            Flag indicating if numba is available. (default False)
        self.verbose (int):
            Verbosity level for print statements. (default 0)
        self.tkinter_widgets (list):
            List of tkinter widgets for the GUI. (default [None, None, None])

    """
    def __init__(self, example_file=None, mass_list_dir=None, tol_MS1=10, tol_MS1_u='ppm', tol_prec=1, tol_prec_u='mz', tol_frag=10, tol_frag_u='ppm', \
                 tol_mob=0.1, tol_mob_u='μs', h=10, w=10, hw_units='mm', is_MS2 = False, is_mobility=False, normalize_img_sizes = True, \
                 pixels_per_line = "mean", output_file_loc = None, in_jupyter = True, testing = False, gui = False, \
                 save_file_format = "npy", ask_confirmation = True,
                 ):
                 
        self.example_file = example_file
        self.mass_list_dir = mass_list_dir
        self.tol_MS1, self.tol_MS1_u = tol_MS1, tol_MS1_u
        self.tol_prec, self.tol_prec_u = tol_prec, tol_prec_u
        self.tol_frag, self.tol_frag_u = tol_frag, tol_frag_u
        self.tol_mob, self.tol_mob_u = tol_mob, tol_mob_u
        self.h = h
        self.w = w
        self.hw_units = hw_units
        self.dimensions = [h, w, hw_units]
        self.is_MS2 = is_MS2
        self.is_mobility = is_mobility
        self.normalize_img_sizes = normalize_img_sizes
        self.pixels_per_line = pixels_per_line
        self.output_file_loc = output_file_loc
        self.save_file_format = save_file_format
        self.ask_confirmation = ask_confirmation
        self.results = {}

        self.HiddenPrints = HiddenPrints
        self.data_format = None # unnecessary unless using .d files

        self.NpEncoder = NpEncoder
        self.get_confirmation_dialogue = get_confirmation
        self.numba_present = "numba" in sys.modules
        self.gui = gui
        self.testing = testing
        self.in_jupyter = in_jupyter
        self.verbose = 0
        self.tkinter_widgets = [None, None, None]

        self.thermo_scan_filter_string_patterns = re.compile(
            r'^(?P<analyzer>FTMS|ITMS|TQMS|SQMS|TOFMS|SECTOR)?\s*'
            r'(?:\{(?P<segment>\d+),(?P<event>\d+)\})?\s*'
            r'(?P<polarity>\+|-)\s*'
            r'(?P<dataType>p|c)\s*'
            r'(?P<source>EI|CI|FAB|ESI|APCI|NSI|TSP|FD|MALDI|GD)?\s*'
            r'(?P<corona>!corona|corona)?\s*'
            r'(?P<photoIonization>!pi|pi)?\s*'
            r'(?P<sourceCID>!sid|sid=-?\d+(?:\.\d+))?\s*'
            r'(?P<detectorSet>!det|det=\d+(?:\.\d+))?\s*'
            r'(?:cv=(?P<compensationVoltage>-?\d+(?:\.\d+)?))?\s*'
            r'(?P<rapid>!r|r)?\s*'
            r'(?P<turbo>!t|t)?\s*'
            r'(?P<enhanced>!e|e)?\s*'
            r'(?P<sps>SPS|K)?\s*'
            r'(?P<dependent>!d|d)?\s*'
            r'(?P<wideband>!w|w)?\s*'
            r'(?P<ultra>!u|u)?\s*'
            r'(?P<supplementalActivation>sa)?\s*'
            r'(?P<accurateMass>!AM|AM|AMI|AME)?\s*'
            r'(?P<scanType>FULL|SIM|SRM|CRM|Z|Q1MS|Q3MS)?\s*'
            r'(?P<lockmass>lock)?\s*'
            r'(?P<multiplex>msx)?\s*'
            r'(?P<msMode>pr|ms|cnl)(?P<msLevel>\d+)?\s*'
            r'(?P<precursorMz>\d+(?:\.\d+)?)?\s*'
            r'(?:@(?P<activationType>cid|hcd|etd)?(?P<activationEnergy>-?\d+(?:\.\d+)?))?\s*'
            r'(?:\[(?P<scanRangeStart>\d+(?:\.\d+)?)-(?P<scanRangeEnd>\d+(?:\.\d+)?)\])?',
            re.IGNORECASE)

        # Allows for initialization without providing an example file or mass list file.
        if (example_file is not None):
            self.file_ext = self.get_file_extension(example_file)
            self.get_metadata_and_params()
            if self.in_jupyter and (not self.testing):
                try: self.display_mass_list()
                except: pass

    @staticmethod
    def get_file_extension(example_file):
        """Returns the file extension of the provided example file."""
        if type(example_file) in [list, tuple]:
            example_file = example_file[0]
        
        return os.path.splitext(example_file)[-1].lower()
    
    # TODO: Include the ability to pass parameters to the GUI from this method
    @staticmethod
    def run_GUI():
        """Runs the MSIGen GUI."""
        import MSIGen.GUI
        MSIGen.GUI.run_GUI()

    def get_metadata_and_params(self, **kwargs):
        """
        Initializes or resets the metadata and parameters for the class.
        List of valid keys along with their default value are:
            'tol_MS1':10,
            'tol_MS1_u':'ppm',
            'tol_prec':1,
            'tol_prec_u':'mz',
            'tol_frag':10,
            'tol_frag_u':'ppm',
            'tol_mob':0.1,
            'tol_mob_u':'μs',
            'h':10,
            'w':10,
            'hw_units':'mm',
            'is_MS2':False,
            'is_mobility':False,
            'normalize_img_sizes':True,
            'output_file_loc':None,
            'in_jupyter':True,
            'testing':False,
            'gui':False,
            'pixels_per_line':"mean",
        """
        defaults_dict = {'tol_MS1':10,
                         'tol_MS1_u':'ppm',
                         'tol_prec':1,
                         'tol_prec_u':'mz',
                         'tol_frag':10,
                         'tol_frag_u':'ppm',
                         'tol_mob':0.1,
                         'tol_mob_u':'μs',
                         'h':10,
                         'w':10,
                         'hw_units':'mm',
                         'is_MS2':False,
                         'is_mobility':False,
                         'normalize_img_sizes':True,
                         'output_file_loc':None,
                         'in_jupyter':True,
                         'testing':False,
                         'gui':False,
                         'pixels_per_line':"mean",}
        for key, value in kwargs.items():
            if key not in defaults_dict:
                raise Exception(f"Invalid keyword argument: {key}")
            elif value is None:
                setattr(self, key, defaults_dict[key])
            else:
                setattr(self, key, value)

        # list of file names for each line scan in the experiment in increasing order of increasing line number
        if type(self.example_file) in [list, tuple]:
            self.line_list = self.example_file
        else:
            self.line_list = self.get_line_list()

        # All mass, mobility, and polarity lists
        self.mass_list = self.get_mass_list(self.mass_list_dir, header = 0, sheet_name = 0)
        # all mass and mobility tolerances
        self.tolerances = self.tol_MS1, self.tol_prec, self.tol_frag, self.tol_mob
        self.tolerance_units = self.tol_MS1_u, self.tol_prec_u, self.tol_frag_u, self.tol_mob_u


    # =======================================================
    # Functions for obtaining the line list
    # =======================================================
    def get_line_list(self, example_file = None, display = False):
        """
        Returns a list of file names for each line scan in the experiment in increasing order of line number.

        Args:
            example_file (str):
                The example file name to use for determining the line list.
                All files in the same directory that match the naming scheme and file extension will be included.
                If None, self.example_file will be used.
            display (bool):
                If True, the line list will be printed to the console. (default False)

        Returns:
            line_list (list):
                List of file names for each line scan in the experiment in increasing order of line number.
        """ 

        if example_file is None:
            example_file = self.example_file
        setattr(self, 'example_file', example_file)
        self.name_body, self.name_post = self.segment_filename(file_name=example_file)
        raw_files = self.get_raw_files()
        self.line_list = self.sort_raw_files(raw_files)

        if display:
            if jupyter_prints:
                iPydisplay(self.line_list)
            else:
                print(self.line_list)
        return self.line_list

    def segment_filename(self, file_name=None):
        """
        Segments file_name into the body (everything before the number preceding the file extension) and post (extension) parts.
        If file_name is None, it uses the example_file attribute.
        
        Args:
            file_name (str):
                The file name to segment. If None, self.example_file will be used.

        Returns:
            name_body (str):
                The body of the file name (everything before the number preceding the file extension).
            name_post (str):
                The file extension (including the dot).
        """

        if file_name is None:
            file_name = self.example_file
        setattr(self, 'example_file', file_name)
        # Get the file extension
        name, name_post = os.path.splitext(file_name)

        # determine the length of the line number
        iterator = 0
        for i in name[::-1]:
            if i.isdigit():
                iterator+=1
            else:
                break

        if iterator<1:
            raise Exception('File names must end in a number that identifies the line.')

        # Get the part of the file name that does not contain the line number
        name_body = name[:-iterator]

        return name_body, name_post

    def get_raw_files(self, name_body = None, name_post = None):
        """
        Returns a list of raw files in the directory that match the naming scheme of the example file.
        Args:
            name_body (str):
                The body of the file name to match (the absolute path except for numbers at the end of the file name). 
                If None, self.name_body will be used.
            name_post (str):
                The file extension to match (including the dot). If None, self.name_post will be used.
        
        Returns:
            raw_files (list):
                List of raw files in the directory that match the naming scheme of the example file.
        """
        if name_body is None:
            name_body = self.name_body
        if name_post is None:
            name_post = self.name_post
        setattr(self, 'name_body', name_body)
        setattr(self, 'name_post', name_post)

        # list all files in directory as absolute paths
        directory = os.path.split(name_body)[0]
        files_in_dir = [os.path.join(directory,i) for i in os.listdir(directory)]

        # remove all files that do not fit the same naming scheme as given file
        raw_files = []
        for file in files_in_dir:
            if file.startswith(name_body) and file.endswith(name_post):
                if file[len(name_body):-len(name_post)].isdigit():
                    raw_files.append(file)

        return raw_files

    def sort_raw_files(self, raw_files, name_body = None, name_post = None):
        """
        Sorts the raw files in ascending order based on their line numbers.
        The line numbers are extracted from the file names by removing the name_body and name_post parts.

        Args:
            raw_files (list):
                List of raw files to sort.
            name_body (str):
                The body of the file name to match (the absolute path except for numbers at the end of the file name).
                If None, self.name_body will be used.
            name_post (str):
                The file extension to match. If None, self.name_post will be used.
        
        Returns:
            sorted_raw_files (list):
                List of raw files sorted in ascending order based on their line numbers.
        """
        if name_body is None:
            name_body = self.name_body
        if name_post is None:
            name_post = self.name_post
        setattr(self, 'name_body', name_body)
        setattr(self, 'name_post', name_post)

        # array of the number of each line file name
        line_nums = [file.replace(name_body,'').replace(name_post,'') for file in raw_files]
        # remove all lines with non-numeric characters (There shouldn't be any, but kept just in case)
        line_nums = np.array([int(line_num) for line_num in line_nums if line_num.isnumeric()])

        # sort the line by their numbers in ascending order
        sorted_raw_files = []
        for i in np.argsort(line_nums):
            sorted_raw_files.append(raw_files[i])
        return sorted_raw_files
    
    # =======================================================
    # Mass list functions
    # =======================================================
    def get_mass_list(self, mass_list_dir = None, header = 0, sheet_name = 0):
        """
        Reads the mass list file and returns a DataFrame containing the mass list.

        Args:
            mass_list_dir (str):
                The directory containing the mass list file. If None, self.mass_list_dir will be used.
            header (int):
                The row number to use as the header in the spreadsheet. (default 0)
            sheet_name (str or int):
                The name (if str) or index (if int) of the sheet to read. (default 0)

        Returns:
            mass_list (list of lists):
                List containing mass/mobility lists split based on MS level.
                Each sublist contains the mass, mobility, or polarity values for that MS level.
        """
        if mass_list_dir is None:
            mass_list_dir = self.mass_list_dir
        setattr(self, 'mass_list_dir', mass_list_dir)

        # read excel-style files
        if mass_list_dir.split('.')[-1].lower() in ['xls','xlsx','xlsm','xlsb','odf','ods','odt']:
            self.raw_mass_list = pd.read_excel(mass_list_dir, header = header, sheet_name = sheet_name)

        # read csv files
        elif mass_list_dir.lower().endswith('csv'):
            self.raw_mass_list = pd.read_csv(mass_list_dir, header = header)
        
        else:
            raise Exception("Transition/Mass list file must be one of the following formats: 'xls','xlsx','xlsm','xlsb','odf','ods','odt','csv'")

        self.raw_mass_list_col_type, self.raw_mass_list_col_idxs = self.column_header_check()
        
        self.mass_list = self.get_mass_mobility_lists()

        return self.mass_list

    def column_header_check(self, raw_mass_list = None):
        """
        Ensures that the column headers include a valid combination of mass, precursor, fragment, mobility, and polarity columns.

        Args:
            raw_mass_list (DataFrame):
                The DataFrame containing the raw mass list. If None, self.raw_mass_list will be used.
        
        Returns:
            raw_mass_list_col_type (list):
                List of column types for the mass list.
            raw_mass_list_col_idxs (list):
                List of indices for the selected mass list columns.
        """
        if raw_mass_list is None:
            raw_mass_list = self.raw_mass_list
        setattr(self, 'raw_mass_list', raw_mass_list)

        # Ensure the column headers arent data values
        if 'float' in str(type(raw_mass_list.columns[0])):
            raise Exception("Ensure that the mass list sheet has column headers. \n\
            For example: 'precursor', 'fragment', 'mobility'")
        
        raw_mass_list_col_type = self.column_type_check()

        raw_mass_list_col_type, raw_mass_list_col_idxs = self.select_mass_list_cols_to_use(raw_mass_list_col_type)

        # This combination is not compatible
        if set(['MS1', 'precursor', 'fragment']).issubset(raw_mass_list_col_type):
            raise Exception("There must be at most two columns containing m/z values, \n\
    one for the precursor m/z and one for the fragment m/z.")
        
        # Ensure at least one column has m/z values
        elif not len([i for i in ['MS1', 'precursor', 'fragment'] if i in raw_mass_list_col_type]):
            raise Exception("There must be at least one column with m/z values \n\
    with one of the following a column headers: 'm/z', 'fragment', 'precursor'")
        
        # if there is MS1 and precursor or fragment columns together, convert MS1 to the correct corresponding column type
        if set(['MS1', 'fragment']).issubset(raw_mass_list_col_type):
            for i, column in enumerate(raw_mass_list_col_type):
                if column == 'MS1':
                    raw_mass_list_col_type[i] = 'precursor'
                    break
        elif set(['MS1', 'precursor']).issubset(raw_mass_list_col_type):
            for i, column in enumerate(raw_mass_list_col_type):
                if column == 'MS1':
                    raw_mass_list_col_type[i] = 'fragment'
                    break
        elif not set(['MS1']).issubset(raw_mass_list_col_type) and (set(['precursor']).issubset(raw_mass_list_col_type) ^ set(['fragment']).issubset(raw_mass_list_col_type)):
            for i, column in enumerate(raw_mass_list_col_type):
                if column in ['precursor', 'fragment']:
                    raw_mass_list_col_type[i] = 'MS1'
                    break
        self.raw_mass_list_col_type, self.raw_mass_list_col_idxs = raw_mass_list_col_type, raw_mass_list_col_idxs
        return raw_mass_list_col_type, raw_mass_list_col_idxs

    def column_type_check(self, raw_mass_list = None):
        """
        Checks the column headers of the mass list and returns a list of column types. (m/z, precursor, fragment, mobility, or polarity)

        Args:
            raw_mass_list (DataFrame):
                The DataFrame containing the raw mass list. If None, self.raw_mass_list will be used.
        
        Returns:
            col_type (list):
                List of column classifications in the mass list based on the column header .
        """
        if raw_mass_list is None:
            raw_mass_list = self.raw_mass_list
        setattr(self, 'raw_mass_list', raw_mass_list)

        # define the header keywords to search for
        kwds = {'MS1_mz_kwds': ['mz','m/z','mass','masses', 'exactmass', 'exactmasses', 'exact_mass', 'exact_masses'],
        'precursor_kwds'     : ['parent','parents','precursor','precursors','prec'],
        'fragment_kwds'      : ['child','fragment','fragments','frag'],
        'mobility_kwds'      : ['mobility','mob','ccs', 'dt', 'drifttime', 'drifttime(ms)', 'collision cross section', 'dt(ms)'],
        'polarity_kwds'      : ['polarity', 'charge']}

        col_type=[]

        # check through each keyword for column type and append to list
        for i, col_name in enumerate(raw_mass_list.columns):
            # remove whitespace
            col_name = "".join(col_name.split())
            if type(col_name) != str:
                col_type.append(None)
            elif col_name.lower() in kwds['MS1_mz_kwds']:
                col_type.append('MS1')
            elif col_name.lower() in kwds['precursor_kwds']:
                col_type.append('precursor')
            elif col_name.lower() in kwds['fragment_kwds']:
                col_type.append('fragment')
            elif col_name.lower() in kwds['mobility_kwds']:
                col_type.append('mobility')
            elif col_name.lower() in kwds['polarity_kwds']:
                col_type.append('polarity')
            else:
                col_type.append(None)
        
        return col_type
    
    def select_mass_list_cols_to_use(self, col_types):
        """
        Filters the column types to include only the valid ones and returns the filtered list and their indices.

        Args:
            col_types (list):
                List of column types to filter.
        
        Returns:
            col_types_filtered (list):
                List of column types with duplicates and columns with NoneType classifications removed.
            col_idxs (list):
                List of indices corresponding to the columns in col_types_filtered.
        """
        col_types_filtered = []
        col_idxs = []

        for i, col_type in enumerate(col_types):
            if col_type not in col_types_filtered and not None:
                col_types_filtered.append(col_type)
                col_idxs.append(i)

        return col_types_filtered, col_idxs
    
    def get_mass_mobility_lists(self, raw_mass_list = None, col_type = None, col_idxs = None):
        """
        Returns the a list containing mass and mobility, and any other values such as polarity, 
        from the raw mass list DataFrame based on col_type and col_idxs.

        Args:
            raw_mass_list (DataFrame):
                The DataFrame containing the raw mass list. If None, self.raw_mass_list will be used.
            col_type (list):
                List of column types to use. If None, self.raw_mass_list_col_type will be used.
            col_idxs (list):
                List of indices for the selected mass list columns. If None, self.raw_mass_list_col_idxs will be used.
        
        Returns:
            self.mass_list (list):
                List containing mass/mobility lists split based on MS level.
                Each sublist contains the mass, mobility, or polarity values for that MS level.
        """
        if raw_mass_list is None:
            raw_mass_list = self.raw_mass_list
        if col_type is None:
            col_type = self.raw_mass_list_col_type
        if col_idxs is None:
            col_idxs = self.raw_mass_list_col_idxs
        setattr(self, 'raw_mass_list', raw_mass_list)
        setattr(self, 'raw_mass_list_col_type', col_type)
        setattr(self, 'raw_mass_list_col_idxs', col_idxs)

        mass_list = raw_mass_list.iloc[:,col_idxs]
        mass_list.columns = col_type
        
        if 'MS1' in col_type:
            MS1_list = mass_list['MS1']
            missing_mass_mask = MS1_list.index[~MS1_list.isna().values | (MS1_list>0).values]
            MS1_list = MS1_list[missing_mass_mask].values.flatten()

            # Save original mass list indices
            mass_list_idxs = [np.array(range(MS1_list.shape[0])).tolist(),[]]

            if 'mobility' in col_type:
                MS1_mob_list = mass_list['mobility']
                MS1_mob_list = MS1_mob_list[missing_mass_mask].fillna(0.0).values.flatten()
            else:
                MS1_mob_list = np.zeros(MS1_list.shape, dtype = np.float64).flatten()
            
            if 'polarity' in col_type:
                MS1_polarity_list = mass_list['polarity']
                MS1_polarity_list = MS1_polarity_list.replace('-',-1.0).replace('+',1.0).fillna(0.0)
                MS1_polarity_list = MS1_polarity_list[missing_mass_mask].values.flatten()
            else:
                MS1_polarity_list = np.zeros(MS1_list.shape, dtype = np.float64).flatten()

            prec_list=np.zeros(0,np.float64)
            frag_list=np.zeros(0,np.float64)
            MS2_mob_list=np.zeros(0,np.float64)
            MS2_polarity_list=np.zeros(0,np.float64)

        else:
            prec_list = mass_list['precursor']
            frag_list = mass_list['fragment']

            if 'mobility' in col_type:
                mob_list = mass_list['mobility']
            else:
                mob_list = pd.DataFrame(np.zeros(prec_list.shape, dtype = np.float64))

            if 'polarity' in col_type:
                polarity_list = mass_list['polarity']
                polarity_list = polarity_list.replace('-',-1.0).replace('+',1.0).fillna(0.0)
            else:
                polarity_list = pd.DataFrame(np.zeros(prec_list.shape, dtype = np.float64))
            
            missing_mass_mask = (prec_list.isna().values|(prec_list<=0).values) & (frag_list.isna().values|(prec_list<=0).values)

            prec_list=prec_list[~missing_mass_mask]
            frag_list=frag_list[~missing_mass_mask]
            mob_list = mob_list[~missing_mass_mask]
            polarity_list = polarity_list[~missing_mass_mask]

            MS1_mask = (prec_list.isna().values | (prec_list<=0).values)^(frag_list.isna().values | (frag_list<=0).values)

            potential_MS1_list1 = prec_list[MS1_mask]
            potential_MS1_list2 = frag_list[MS1_mask]

            MS1_list = pd.concat((potential_MS1_list1, potential_MS1_list2)).sort_index().dropna().values.flatten()
            MS1_list = MS1_list[np.where(MS1_list>0)]
            MS1_mob_list = mob_list[MS1_mask].fillna(0.0).values.flatten()
            MS1_polarity_list = polarity_list[MS1_mask].fillna(0.0).values.flatten()

            # Save original mass list indices for later reordering
            mass_list_range = np.array(range(prec_list.values.flatten().shape[0]))
            mass_list_idxs = [mass_list_range[MS1_mask].tolist(), mass_list_range[~MS1_mask].tolist()]

            prec_list=prec_list[~MS1_mask].values.flatten()
            frag_list=frag_list[~MS1_mask].values.flatten()
            MS2_mob_list = mob_list[~MS1_mask].fillna(0.0).values.flatten()
            MS2_polarity_list = polarity_list[~MS1_mask].fillna(0.0).values.flatten()

        self.mass_list = MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs
        self.MS1_list, self.MS1_mob_list, self.MS1_polarity_list, self.prec_list, self.frag_list, self.MS2_mob_list, self.MS2_polarity_list, self.mass_list_idxs = self.mass_list

        # get a mass list in the same order as pixels and save it to the metadata dict
        self.final_mass_list = [['TIC']]+[None]*(len(mass_list_idxs[0])+len(mass_list_idxs[1]))
        for counter, idx in enumerate(mass_list_idxs[0]):
            self.final_mass_list[idx+1]=[MS1_list[counter], MS1_mob_list[counter], MS1_polarity_list[counter]]
        for counter, idx in enumerate(mass_list_idxs[1]):
            self.final_mass_list[idx+1]=[prec_list[counter], frag_list[counter], MS2_mob_list[counter], MS2_polarity_list[counter]]

        return self.mass_list

    def display_mass_list(self):
        """Displays the mass list as a DataFrame with appropriate formatting."""
        mass_list = deepcopy(self.final_mass_list)
        a = mass_list[1:]
        for i in range(len(a)):
            # add in fragment if needed
            if len(a[i])==3:
                a[i][1:1] = [0.0]

            # convert polarity to symbol
            pol = a[i][3]
            if pol > 0:
                a[i][3] = '+'
            elif pol < 0:
                a[i][3] = '-'
            elif pol == 0:
                a[i][3] = ''

        columns = ["Precursor", "Fragment", "Mobility", "Polarity"]
        _display_df(pd.DataFrame(a, columns =columns,  index = range(1, len(a)+1)))

    # =======================================================
    # Functions for obtaining m/z and mobility windows
    # =======================================================
    def get_mass_or_mobility_window(self, val_list, tol, unit):
        """
        Determines the upper and lower bounds for a selection window based on the provided values, tolerance.
        Defines the lower_lims and upper_lims attributes of the class.

        Args:
            val_list (list):
                List of values for which to determine the selection window.
                These values define the center of the selection window.
            tol (float):
                Tolerance value for the selection window. The window will be +/- this value.
            unit (str):
                Units for the tolerance (ex. 'ppm', 'mz', etc).
        """
        if self.lower_lims is None:
            self.lower_lims = []
        if self.upper_lims is None:
            self.upper_lims = []

        if unit.lower() == 'ppm':
            self.lower_lims.append(np.clip(np.array(val_list) * (1-(tol/1000000)), 0, None))
            self.upper_lims.append(np.array(val_list) * (1+(tol/1000000)))
        else:
            self.lower_lims.append(np.clip((np.array(val_list) - tol), 0, None))
            self.upper_lims.append(np.array(val_list) + tol)

    def get_all_ms_and_mobility_windows(self, mass_lists=None, tolerances=None, tolerance_units=None):
        """
        Determines the upper and lower bounds for each mass or mobility window based on the provided mass lists, tolerances, and units.

        Args:
            mass_lists (list):
                List of mass lists to use for the selection windows. If None, self.mass_list will be used.
            tolerances (list):
                List of tolerances to use for the selection windows. If None, self.tolerances will be used.
            tolerance_units (list):
                List of units for the tolerances. If None, self.tolerance_units will be used.
        
        returns:
            lower_lims (list of arrays):
                The lower limit of each selection window.
            upper_lims (list of arrays):
                The upper limit of each selection window.
        """
        if mass_lists is None:
            mass_lists = self.mass_list
        if tolerances is None:
            tolerances = self.tolerances
        if tolerance_units is None:
            tolerance_units = self.tolerance_units
        setattr(self, 'mass_list', mass_lists)
        setattr(self, 'tolerances', tolerances)
        setattr(self, 'tolerance_units', tolerance_units)
        
        # unpack variables
        self.MS1_list, self.MS1_mob_list, self.MS1_polarity_list, self.prec_list, self.frag_list, self.MS2_mob_list, self.MS2_polarity_list, self.mass_list_idxs = self.mass_list
        self.tol_MS1, self.tol_prec, self.tol_frag, self.tol_mob = self.tolerances
        self.tol_MS1_u, self.tol_prec_u, self.tol_frag_u, self.tol_mob_u = self.tolerance_units

        self.upper_lims = []
        self.lower_lims = []

        # get upper and lower limits in order:
        # MS1, MS1_mob, MS1_polarity, precursor, fragment, MS2_mob, MS2_polarity
        self.get_mass_or_mobility_window(self.MS1_list, self.tol_MS1, self.tol_MS1_u)
        self.get_mass_or_mobility_window(self.MS1_mob_list, self.tol_mob, self.tol_mob_u)
        self.lower_lims.append(np.array(self.MS1_polarity_list))
        self.upper_lims.append(np.array(self.MS1_polarity_list))
        self.get_mass_or_mobility_window(self.prec_list, self.tol_prec, self.tol_prec_u)
        self.get_mass_or_mobility_window(self.frag_list, self.tol_frag, self.tol_frag_u)
        self.get_mass_or_mobility_window(self.MS2_mob_list, self.tol_mob, self.tol_mob_u)
        self.lower_lims.append(np.array(self.MS2_polarity_list))
        self.upper_lims.append(np.array(self.MS2_polarity_list))

        return self.lower_lims, self.upper_lims
    
    # ============================================
    # Main Data Extraction Workflow Functions
    # ============================================
    def get_image_data(self, **kwargs):
        """
        Processes the image data for the specified mass list and line list.
        Saves and returns the processed image data.
        Requires a subclass to run successfully.
        All arguments are optional and will update the corresponding class attribute values.
        Accepted arguments are: verbose, in_jupyter, testing, gui, results, pixels_per_line, tkinter_widgets
        
        Returns:
            metadata (dict):
                Dictionary containing metadata about the image data.
            pixels (np.array or list):
                A 3D array or list of pixel image data extracted from the image.
        """
        invalid_keys = []
        premissible_keys = ['verbose', 'in_jupyter', 'testing', 'gui', 'results', 'pixels_per_line', \
                            'tkinter_widgets', 'save_file_format', 'ask_confirmation']
        for key, value in kwargs.items():
            if key not in premissible_keys:
                invalid_keys.append(key)
                continue
            setattr(self, key, value)

        # Provides list of invalid keys if any are present
        if len(invalid_keys) > 0:
            raise Exception(f"Invalid keyword argument: {[key for key in invalid_keys]}")
        
        self.lower_lims, self.upper_lims = self.get_all_ms_and_mobility_windows()

        self.metadata = self.make_metadata_dict()

        self.metadata, self.pixels = self.load_files(metadata = self.metadata)

        if self.gui:
            self.results["metadata"] = self.metadata
            self.results["pixels"] = self.pixels

        self.save_pixels(self.metadata, self.pixels, file_format=self.save_file_format, ask_confirmation=self.ask_confirmation)

        return self.metadata, self.pixels

    def load_files(self, *args, **kwargs):
        """Implemented in subclasses."""
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")

    def get_scan_without_zeros(self, *args, **kwargs):
        """Implemented in subclasses."""
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms1_no_mob(self, *args, **kwargs):
        """Implemented in subclasses."""
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms2_no_mob(self, *args, **kwargs):
        """Implemented in subclasses."""
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms1_mob(self, *args, **kwargs):
        """Implemented in subclasses."""
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms2_mob(self, *args, **kwargs):
        """Implemented in subclasses."""
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def check_dim(self, *args, **kwargs):
        """Implemented in subclasses."""
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    # ============================================
    # GUI progressbar methods
    # ============================================
    def progressbar_start_preprocessing(self):
        """Displays the progress bar while preprocessing the data."""
        if self.gui:
            self.tkinter_widgets[1]['text']="Preprocessing data"
            self.tkinter_widgets[1].update()

    def progressbar_start_extraction(self):
        """Displays the progress bar showing completion of data processing."""
        if self.gui:
            self.tkinter_widgets[1]['text']="Extracting data"
            self.tkinter_widgets[1].update()

    def progressbar_update_progress(self, num_spe, i, j):
        """
        Updates the progress bar with the current progress.
        
        Args:
            num_spe (int):
                The number of spectra in the current line scan.
            i (int):
                The current line number being processed.
            j (int):
                The current spectrum number being processed.
        """            
        if self.gui:
            self.tkinter_widgets[0]['value']=(100*i/len(self.line_list))+((100/len(self.line_list))*(j/num_spe))
            self.tkinter_widgets[0].update()
            self.tkinter_widgets[2]['text']=f'line {i+1}/{len(self.line_list)}, spectrum {j+1}/{num_spe}'
            self.tkinter_widgets[2].update()



    # ============================================
    # General Data Processing Functions
    # ============================================

    # ============================================
    # Slicing sorted data for mass selection 
    # ============================================
    def extract_masses_no_mob(self, mz, lb, ub, intensity_points):
        """
        Finds all values of mz within the mass windows defined by lower bounds lb and upper bounds ub and
        returns the summed intensities of those m/z values. 
        The provided data must be sorted form lowest m/z to greatest m/z and not contain mobility information.
        Length of l and r must be the same.

        Args:
            mz (np.ndarray):
                The m/z values to search through.
            lb (list):
                The lower bounds of the mass windows.
            ub (float):
                The upper bounds of the mass windows.
            intensity_points (np.ndarray):
                The intensity values corresponding to each m/z value.

        Returns:
            pixel (np.ndarray):
                The summed intensity values for each mass window.
        """
        if self.numba_present:
            idxs_to_sum = self.vectorized_sorted_slice_njit(mz, lb, ub)
            pixel = self.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
        else:
            idxs_to_sum = self.vectorized_sorted_slice(mz, lb, ub) # Usually slower
            pixel = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)
        return pixel

    @staticmethod
    def sorted_slice(a,l,r):
        """
        Outputs the indices where the values of numpy array (a) are within a given lower (l) and upper (r) bound.
        Array (a) must be in order of increasing value for this to be used.

        Args:
            a (np.ndarray):
                The array to search through.
            l (float):
                The lower bound of the mass window.
            r (float):
                The upper bound of the mass window.

        Returns:
            np.array:
                The indices of the values in the array that are within the given bounds.
        """
        start = np.searchsorted(a, l, 'left')
        end = np.searchsorted(a, r, 'right')
        # print(np.arange(start,end))
        return np.arange(start,end)

    @staticmethod
    def vectorized_sorted_slice(a,l,r):
        """
        Outputs a list of indices where the values of numpy array (a) are within a given 
        lower and upper bounds for for each entry in the vectors containing (l) and upper (r) bounds.
        Array (a) must be in order of increasing value for this to be used.
        Length of l and r must be the same.

        Args:
            a (np.ndarray):
                The array to search through.
            l (float):
                The lower bounds of the mass windows.
            r (float):
                The upper bounds of the mass windows.
        
        Returns:
            np.array:
                A 2D array of indices where the values of the array are within the given bounds.
                Each row corresponds to a mass window defined by the lower and upper bounds.
                Each column corresponds to an index in the array that is within the bounds.
        """
        start = np.searchsorted(a, l, 'left')
        end = np.searchsorted(a, r, 'right')
        ls = [list(range(start[i],end[i])) for i in range(len(start))]
        max_num_idxs = max([len(i) for i in ls])
        if not max_num_idxs:
            max_num_idxs = 1
        ls = [i+[-1]*(max_num_idxs-len(i)) for i in ls]
        return np.array(ls)

    @staticmethod
    def vectorized_unsorted_slice(mz,lbs,ubs):
        """
        Outputs a list of indices where the values of numpy array (a) are within a given 
        lower and upper bounds for for each entry in the vectors containing (l) and upper (r) bounds.
        Works with unsorted arrays.

        Args:
            mz (np.ndarray):
                The array to search through.
            lbs (float):
                The lower bounds of the mass windows.
            ubs (float):
                The upper bounds of the mass windows.
        
        Returns:
            np.array:
                A 2D array of indices where the values of the array are within the given bounds.
                Each row corresponds to a mass window defined by the lower and upper bounds.
                Each column corresponds to an index in the array that is within the bounds.
        """
        mass_idxs, int_idxs = np.where((mz[None,:]>lbs[:,None])&(mz[None,:]<ubs[:,None]))
        ls = [int_idxs[mass_idxs == i].tolist() for i in range(len(lbs))]
        max_num_idxs = max([len(i) for i in ls])
        if not max_num_idxs:
            max_num_idxs = 1
        ls = [i+[-1]*(max_num_idxs-len(i)) for i in ls]
        return np.array(ls)

    @staticmethod
    def vectorized_unsorted_slice_mob(mz,mob,lbs,ubs,mob_lbs,mob_ubs):
        """
        Outputs a list of indices where the values of the m/z array (mz) are within a given 
        lower and upper bounds for for each entry in the vectors containing lower (lbs) and upper (rbs) bounds
        and where the values of the mobility array (mob) are within a given 
        lower and upper bounds for for each entry in the vectors containing lower (mob_lbs) and upper (mob_rbs) bounds.
        Works with unsorted arrays.

        Args:
            mz (np.ndarray):
                The array to search through.
            mob (np.ndarray):
                The mobility array to search through.
            lbs (np.ndarray):
                The lower bounds of the mass windows.
            ubs (np.ndarray):
                The upper bounds of the mass windows.
            mob_lbs (np.ndarray):
                The lower bounds of the mobility windows.
            mob_ubs (np.ndarray):
                The upper bounds of the mobility windows.

        Returns:
            np.array:
                A 2D array of indices where the values of the array are within the given bounds.
                Each row corresponds to a mass/mobility window defined by the lower and upper bounds.
                Each column corresponds to an index in the array that is within the bounds.
        """
        mass_idxs, int_idxs = np.where((mz[None,:]>lbs[:,None])&(mz[None,:]<ubs[:,None])&(mob[None,:]>mob_lbs[:,None])&(mob[None,:]<mob_ubs[:,None]))
        ls = [int_idxs[mass_idxs == i].tolist() for i in range(len(lbs))]
        max_num_idxs = max([len(i) for i in ls])
        if not max_num_idxs:
            max_num_idxs = 1
        ls = [i+[-1]*(max_num_idxs-len(i)) for i in ls]
        return np.array(ls)

    # Since numba is an optional package and these imports will fail if njit is not imported from numba, these are only defined if it's present
    
    def vectorized_sorted_slice_njit(self,a,l,r):
        """
        Outputs a list of indices where the values of numpy array (a) are within a given 
        lower and upper bounds for for each entry in the vectors containing (l) and upper (r) bounds.
        Array (a) must be in order of increasing value for this to be used.
        Length of l and r must be the same.
        If numba is not present, this will run vectorized_sorted_slice instead.

        Args:
            a (np.ndarray):
                The array to search through.
            l (float):
                The lower bounds of the mass windows.
            r (float):
                The upper bounds of the mass windows.
        
        Returns:
            np.array:
                A 2D array of indices where the values of the array are within the given bounds.
                Each row corresponds to a mass window defined by the lower and upper bounds.
                Each column corresponds to an index in the array that is within the bounds.
        """

        if "numba" in sys.modules:
            return self._vectorized_sorted_slice_njit(a,l,r)
        else:
            return self.vectorized_sorted_slice(a,l,r)

    def assign_values_to_pixel_njit(self, intensities, idxs_to_sum):
        """
        Assigns values to pixels array based on the provided intensities and indices.
        Uses numba njit to speed up the process. 
        Will run the same function without njit if numba is not present.

        Args:
            intensities (np.ndarray):
                The intensity values of the pixel to pixels.
            idxs_to_sum (np.ndarray):
                The indices of the intensity values to sum for each pixel.
        
        Returns:
            np.array:
                The summed intensity values for each pixel.
        """
        if "numba" in sys.modules:
            return self._assign_values_to_pixel_njit(intensities, idxs_to_sum)
        else:
            return np.sum(np.take(intensities, idxs_to_sum), axis = 1)


    if "numba" in sys.modules:
        # Potential improvement to vectorized_sorted_slice using numba
        @staticmethod
        @njit
        def _vectorized_sorted_slice_njit(a,l,r):
            """
            Slices numpy array 'a' based on given vectors of lower (l) and upper (r) bounds.
            Array must be sorted for this to be used.
            Length of l and r must be the same.
            Only defined if numba is imported.
            """
            start = np.searchsorted(a, l, 'left')
            end = np.searchsorted(a, r, 'right')
            # print(start, end)
            ls = [list(range(start[i],end[i])) for i in range(len(start))]
            max_num_idxs = max([len(i) for i in ls])
            if not max_num_idxs:
                max_num_idxs = 1
            arr = -1 * np.ones((len(ls), max_num_idxs), dtype = np.int32)
            for i, j in enumerate(ls):
                arr[i, :len(j)] = j
            return arr

        # gets summed intensity of each mass at a pixel slightly faster
        @staticmethod
        @njit
        def _assign_values_to_pixel_njit(intensities, idxs_to_sum):
            """
            Assigns values to pixels array based on the provided intensities and indices.
            Only defined if numba is imported.
            """
            return np.sum(np.take(intensities, idxs_to_sum), axis = 1)
    
    @staticmethod
    def flatten_list(l):
        """Flattens a nested list into a single list."""
        return [item for sublist in l for item in sublist]

    # ============================================
    # MS1 specific data processing functions
    # ============================================

    def ms1_interp(self, pixels, rts = None, mass_list = None, pixels_per_line = None):
        """
        Interpolates MS1 data to create a 2D image for each entry in the mass list.
        Interpolation is done by normalizing retention times of each line to be between 0 and 1.
        A 2D grid is created with the specified height (number of line) and width (pixels per line) 
        and the data is interpolated onto this grid using nearest-neighbor interpolation.
        
        Args:
            pixels (np.ndarray): 
                List of arrays of shape (pixels_per_line, m/z) containing intenisty data.
                Each entry represents a single line scan.
            rts (list of np.array): 
                (optional) List of retention times for each line. If None, uses the class attribute self.rts.
            mass_list (np.ndarray): 
                (optional) 2D array of shape (m/z, 1) containing the mass list. If None, uses self.MS1_list.
            pixels_per_line (str or int): 
                (optional) Number of pixels per line for the output image. If None, uses self.pixels_per_line.
                Valid options are "min", "max", "mean", or an integer.

        Returns:
            pixels_aligned (np.ndarray): 
                3D array of shape (m/z+1, lines, pixels_per_line) containing the interpolated data.
                The last dimension contains the mass list and the intensity data for each pixel.
        """
        if pixels_per_line is not None:
            self.pixels_per_line = pixels_per_line
        if rts is not None:
            self.rts = rts
        if mass_list is None:
            mass_list = self.MS1_list

        # normalize the retention times to be [0-1] and find evenly spaced times to resample at
        rts_normed = [(line_rts - line_rts.min())/(line_rts.max() - line_rts.min()) for line_rts in self.rts]
        if self.pixels_per_line == "mean":
            rts_aligned = np.linspace(0, 1, int(np.mean([len(rts) for rts in rts_normed])))
        elif self.pixels_per_line == "max":
            rts_aligned = np.linspace(0, 1, int(np.max([len(rts) for rts in rts_normed])))
        elif self.pixels_per_line == "min":
            rts_aligned = np.linspace(0, 1, int(np.min([len(rts) for rts in rts_normed])))
        elif type(pixels_per_line) == int:
            rts_aligned = np.linspace(0, 1, pixels_per_line)
        else:
            raise ValueError("pixels_per_line must be either 'mean', 'max', 'min', or an integer")

        # Initialize pixels
        pixels_aligned = np.empty([len(self.line_list), len(rts_aligned), (mass_list.shape[0]+1)])

        # Interpolate each line with nearest neighbor to align number of pixels per line
        X = np.arange(pixels_aligned.shape[-1])
        for idx, line in enumerate(pixels):
            coords = (rts_normed[idx], X)
            line_pixels_aligned = interpn(coords, line, np.array(np.meshgrid(rts_aligned,X)).reshape(2, -1).transpose(1,0), method='nearest').reshape(X.size, rts_aligned.size)
            pixels_aligned[idx] = line_pixels_aligned.T
            
        # makes axes (m/z, h, w)
        pixels_aligned = np.moveaxis(pixels_aligned, -1, 0)
        
        return pixels_aligned

    # ============================================
    # MS2 specific data processing functions
    # ============================================
    def ms2_interp(self, pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, mzs_per_filter_grp, normalize_img_sizes = None, pixels_per_line = None):
        """
        Interpolates MS2 data to create a 2D image for each entry in the mass list.
        Interpolation is done by normalizing retention times of each line to be between 0 and 1.
        If normalize_img_sizes is True, the interpolation is the same as in ms1_interp.
        If normalize_img_sizes is False, each filter group is independently interpolated, resulting in images of varying size stored in a list.
        
        Args:
            pixels_metas (list): 
                A list of lists of 2D arrays of shape (pixels_per_line, # of m/z in the group) containing intenisty data.
                Each entry represents a single line scan that is made up of a list representing each group of transitions.
            all_TimeStamps (list of np.array): 
                A nested list containing retention times for each pixel in pixels_meta.
            acq_times (list of np.array):
                A nested list containing the acquisition times for all spectra in each line scan, whether used or not.
            scans_per_filter_grp (list):
                A list of lists containing the number of spectra per filter group for each line scan.
            mzs_per_filter_grp (list):
                A list of lists containing the m/z values for each filter group for each line scan.
            normalize_img_sizes (bool):
                (optional) If True, all images will be resized to the same size, being the maximum number of spectra per filter group.
            pixels_per_line (str or int): 
                (optional) Number of pixels per line for the output images. If None, uses self.pixels_per_line.
                Valid options are "min", "max", "mean", or an integer
        """
        if pixels_per_line is not None:
            self.pixels_per_line = pixels_per_line
        if normalize_img_sizes is not None:
            self.normalize_img_sizes = normalize_img_sizes

        # Normalize timestamps to align each line in case one line took longer or started later.
        all_TimeStamps_normed  = self.normalize_ms2_timestamps(all_TimeStamps, acq_times)

        # Deterime how many pixels to use for each group of transitions and get evenly spaced times to sample at
        num_spe_per_group_aligned = self.get_num_spe_per_group_aligned(scans_per_filter_grp)
        # print(num_spe_per_group_aligned)
        all_TimeStamps_aligned = [np.linspace(0,1,i) for i in num_spe_per_group_aligned]

        # make the final output of shape (lines, pixels_per_line, num_transitions+1)
        pixels = [np.zeros((len(self.line_list), num_spe_per_group_aligned[i], len(mzs)+1)) for (i, mzs) in enumerate(mzs_per_filter_grp)]
        # print(num_spe_per_group_aligned)
        # go through the extracted data and place them into pixels_final. list by group idx with shapes (# of lines, # of Pixels per line, m/z)
        for i, pixels_meta in enumerate(pixels_metas):
            for j, pixels_meta_grp in enumerate(pixels_meta):
                points = (all_TimeStamps_normed[i][j], np.arange(pixels_meta_grp.shape[1]))
                sampling_points = np.array(np.meshgrid(*(all_TimeStamps_aligned[j], np.arange(pixels_meta_grp.shape[1])), indexing = 'ij')).transpose(1,2,0)                    
                # Ensure things work even if there are no points in the group
                if points[0].shape[0] == 0:
                    points = (np.array([0]), np.arange(pixels_meta_grp.shape[1]))
                    pixels_meta_grp = np.zeros((1, pixels_meta_grp.shape[1]))
                # print(points[0].shape, points[1].shape, pixels_meta_grp.shape, sampling_points.shape, j, i)
                pixels[j][i] = interpn(points, pixels_meta_grp, sampling_points, method = 'nearest', bounds_error = False, fill_value=None)
        return pixels, all_TimeStamps_aligned
    
    def normalize_ms2_timestamps(self, all_TimeStamps, acq_times):
        """Normalizes the retention times of each line to be between 0 and 1 for MS2 data."""
        all_TimeStamps_normed = []
        for i, line_timestamps in enumerate(all_TimeStamps):
            t_max = max(acq_times[i])
            t_min = min(acq_times[i])
            all_TimeStamps_normed.append([])
            for grp_timestamps in line_timestamps:
                all_TimeStamps_normed[i].append((grp_timestamps-t_min)/(t_max-t_min))
        return all_TimeStamps_normed
    
    ## This is used for D and mzML files, but is overwritten for raw files.
    def get_filters_info(self, all_filters_list):
        """Collects information that would be present in Thermo filters."""
        filter_list = []
        acq_polars = []
        acq_types = []
        precursors = []
        mz_ranges = []
        mob_ranges = []
        for i in all_filters_list:
            filter_list.extend(i)
        filter_list, filter_inverse = np.unique(filter_list, return_inverse=True, axis = 0)
        
        for i in filter_list:
            if self.is_mobility:
                (mz, energy, level, polarity, mass_range_start, mass_range_end, mob_range_start, mob_range_end) = i
            else:
                (mz, energy, level, polarity, mass_range_start, mass_range_end) = i
            
            if polarity == '+':
                p = 1.0
            if polarity == '-':
                p = -1.0
            else:
                p = 0.0
            
            acq_polars.append(p)
            acq_types.append(level)
            precursors.append(mz)
            mz_ranges.append([mass_range_start, mass_range_end])
            if self.is_mobility:
                mob_ranges.append([mob_range_start, mob_range_end])
        
        if self.is_mobility:
            return [filter_list, acq_polars, acq_types, precursors, mz_ranges, mob_ranges], filter_inverse
        else:
            return [filter_list, acq_polars, acq_types, precursors, mz_ranges], filter_inverse
    
    def get_filter_idx(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")

    def get_num_spe_per_group_aligned(self, scans_per_filter_grp, normalize_img_sizes=None, pixels_per_line = None):
        """
        Determines the number of spectra per filter group.
        If normalize_img_sizes is True, all images will be resized to the same size, being the maximum number of spectra per filter group.
        If normalize_img_sizes is False, each filter group is independently resized, resulting in images of varying size.
        """
        if normalize_img_sizes is not None:
            self.normalize_img_sizes = normalize_img_sizes
        if pixels_per_line is not None:
            self.pixels_per_line = pixels_per_line

        if self.pixels_per_line == "mean":
            num_spe_per_group_aligned = np.ceil(np.mean(np.array(scans_per_filter_grp), axis = 0)).astype(int)
        elif self.pixels_per_line == "max":
            num_spe_per_group_aligned = np.ceil(np.max(np.array(scans_per_filter_grp), axis = 0)).astype(int)
        elif self.pixels_per_line == "min":
            num_spe_per_group_aligned = np.ceil(np.min(np.array(scans_per_filter_grp), axis = 0)).astype(int)
        elif type(self.pixels_per_line) == int:
            num_spe_per_group_aligned = np.full(scans_per_filter_grp.shape[1], self.pixels_per_line)
        else:
            raise ValueError("pixels_per_line must be either 'mean', 'max', 'min', or an integer")

        if self.normalize_img_sizes == True:
            num_spe_per_group_aligned = np.full(num_spe_per_group_aligned.shape, num_spe_per_group_aligned.max(), dtype = int)

        return num_spe_per_group_aligned

    def get_ScansPerFilter(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")

    def get_CountsPerFilter(self, filters_info):
        """Gets information about the peaks present in each ms2 filter."""
        # unpack vars
        if len(filters_info) == 5:
            filter_list, acq_polars, acq_types, precursors, mz_ranges = filters_info
            is_mob = False
            mob_ranges = []
        elif len(filters_info) == 6:
            filter_list, acq_polars, acq_types, precursors, mz_ranges, mob_ranges = filters_info
            is_mob = True
        MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs = self.mass_list
        MS1_lb, MS1_mob_lb, _, prec_lb, frag_lb, ms2_mob_lb, _ = self.lower_lims
        MS1_ub, MS1_mob_ub, _, prec_ub, frag_ub, ms2_mob_ub, _ = self.upper_lims

        mzsPerFilter = [ [] for _ in range(filter_list.shape[0]) ]
        mzsPerFilter_lb = [ [] for _ in range(filter_list.shape[0]) ]
        mzsPerFilter_ub = [ [] for _ in range(filter_list.shape[0]) ]
        mzIndicesPerFilter = [ [] for _ in range(filter_list.shape[0]) ]
        mobsPerFilter_lb = [ [] for _ in range(filter_list.shape[0]) ]
        mobsPerFilter_ub = [ [] for _ in range(filter_list.shape[0]) ]

        if MS1_list.shape[0]:
            for i, mz in enumerate(MS1_list):
                list_polarity = MS1_polarity_list[i]
                for j in range(filter_list.shape[0]):
                    acq_type = acq_types[j]
                    mz_range = mz_ranges[j]
                    polarity = acq_polars[j]
                    
                    # Get all data based on whether mobility is being used
                    if not is_mob:
                        if (acq_type in ['Full ms', 'SIM ms', 'MS1']) \
                            and (mz >= float(mz_range[0])) & (mz <= float(mz_range[1])) \
                            and (list_polarity in [0., None, polarity]):

                            mzsPerFilter[j].append(mz)
                            mzsPerFilter_lb[j].append(MS1_lb[i])
                            mzsPerFilter_ub[j].append(MS1_ub[i])
                            mzIndicesPerFilter[j].append(i)
                        
                    else:
                        mob_range = mob_ranges[j]
                        mob = MS1_mob_list[i]
                        if (acq_type in ['Full ms', 'SIM ms', 'MS1']) \
                            and (mz >= float(mz_range[0])) & (mz <= float(mz_range[1])) \
                            and (mob >= float(mob_range[0])) & (mob <= float(mob_range[1])) \
                            and (list_polarity in [0., None, polarity]):

                            mzsPerFilter[j].append(mz)
                            mzsPerFilter_lb[j].append(MS1_lb[i])
                            mzsPerFilter_ub[j].append(MS1_ub[i])
                            mzIndicesPerFilter[j].append(i)
                            mobsPerFilter_lb[j].append(MS1_mob_lb[i])
                            mobsPerFilter_ub[j].append(MS1_mob_ub[i])

        if prec_list.shape[0]:
            for i in range(prec_list.shape[0]):
                list_frag = frag_list[i]
                list_mob = MS2_mob_list[i]
                list_polarity = MS2_polarity_list[i]

                for j in range(filter_list.shape[0]):
                    acq_type = acq_types[j]
                    prec = float(precursors[j])
                    frag_range = mz_ranges[j]
                    polarity = acq_polars[j]
                    
                    if not is_mob:
                        # TODO: Check that "SIM MS2" should not be "SIM ms2" or "sim ms2"
                        if (acq_type in ['Full ms2', 'SIM MS2', 'MS2', 'MRM', 'diaPASEF', 'ddaPASEF']) \
                            and ((prec >= float(prec_lb[i])) & (prec <= float(prec_ub[i]))) \
                            and (list_frag >= float(frag_range[0])) & (list_frag <= float(frag_range[1])) \
                            and (list_polarity in [0., None, polarity]):
                                
                            mzsPerFilter[j].append(list_frag)
                            mzsPerFilter_lb[j].append(frag_lb[i])
                            mzsPerFilter_ub[j].append(frag_ub[i])

                            mzIndicesPerFilter[j].append(i)
                    
                    else:
                        mob_range = mob_ranges[j]
                        # TODO: Check that "SIM MS2" should not be "SIM ms2" or "sim ms2"
                        if (acq_type in ['Full ms2', 'SIM MS2', 'MS2', 'MRM', 'diaPASEF', 'ddaPASEF']) \
                            and ((prec >= float(prec_lb[i])) & (prec <= float(prec_ub[i]))) \
                            and (list_frag >= float(frag_range[0])) & (list_frag <= float(frag_range[1])) \
                            and (list_mob >= float(mob_range[0])) & (list_mob <= float(mob_range[1])) \
                            and (list_polarity in [0., None, polarity]):

                            mzsPerFilter[j].append(list_frag)
                            mzsPerFilter_lb[j].append(frag_lb[i])
                            mzsPerFilter_ub[j].append(frag_ub[i])
                            mzIndicesPerFilter[j].append(i)
                            mobsPerFilter_lb[j].append(ms2_mob_lb[i])
                            mobsPerFilter_ub[j].append(ms2_mob_ub[i])

        for i, mz in enumerate(mzsPerFilter):
            sortmask = np.argsort(mz)
            mzsPerFilter[i] = np.array(mzsPerFilter[i])[sortmask].tolist()
            mzsPerFilter_lb[i] = np.array(mzsPerFilter_lb[i])[sortmask].tolist()
            mzsPerFilter_ub[i] = np.array(mzsPerFilter_ub[i])[sortmask].tolist()
            mzIndicesPerFilter[i] = np.array(mzIndicesPerFilter[i])[sortmask].tolist()
            if is_mob:
                mobsPerFilter_lb[i] = np.array(mobsPerFilter_lb[i])[sortmask].tolist()
                mobsPerFilter_ub[i] = np.array(mobsPerFilter_ub[i])[sortmask].tolist()

        # fill in peaks per filter info using both filter (MRM) and mass list info
        if not is_mob:
            return mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter
        else:
            return mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mobsPerFilter_lb, mobsPerFilter_ub, mzIndicesPerFilter

    def consolidate_filter_list(self, filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter):
        """
        The function will group together MS2 filters that are present in the same scans.
        This is necessary to deals with the case where ms2 filters do not have matching mass ranges, which is common with Agilent data.
        """
        # unpack filters_info
        filters_list, acq_polars, acq_types, precursors = filters_info[:4]

        # Make a boolean array where the filters match:
        # precursors, polarity, mode, and have the same fragments in the mass range
        # bool_arr = np.eye(len(mzsPerFilter), dtype = bool)
        groups = []
        group_prec_bounds = []
        consolidated_filter_list = []
        consolidated_idx_list = []

        for i, mzs_in_filter in enumerate(mzIndicesPerFilter):
            used = False
            for idx, group in enumerate(groups):
                if ([acq_polars[i],acq_types[i], mzs_in_filter] == [group[0], group[1], group[3]]) and \
                    ((precursors[i] > group_prec_bounds[idx][0]) and (precursors[i] < group_prec_bounds[idx][1])):
                    used = True
                    consolidated_filter_list[idx].append(filters_list[i])
                    consolidated_idx_list[idx].append(i)

                    # Update the group with the a new precursor m/z if the values do not match
                    # uses a weighted average of the current precursor and the new one
                    if precursors[i] != group[2]:
                        # use a weighted average of the current precursor (group[2] weighted with the number of ) and the new one
                        groups[idx][2] = (group[2]*(len(consolidated_idx_list[idx])-1) + precursors[i]) / (len(consolidated_idx_list[idx]))
                        # Update the group with the new bounds
                        if self.tol_prec_u.lower() == 'mz':
                            group_prec_bounds[idx] = [groups[idx][2] - self.tol_prec, groups[idx][2] + self.tol_prec]
                        else: #assume ppm
                            group_prec_bounds[idx] = [groups[idx][2] - groups[idx][2]*self.tol_prec/1e6, groups[idx][2] + groups[idx][2]*self.tol_prec/1e6]
                    break

            if used == False:
                # Include a filter 
                if self.tol_prec_u.lower() == 'mz':
                    group_prec_bounds.append([precursors[i] - self.tol_prec, precursors[i] + self.tol_prec])
                else: #assume ppm
                    group_prec_bounds.append([precursors[i] - precursors[i]*self.tol_prec/1e6, precursors[i] + precursors[i]*self.tol_prec/1e6])
                groups.append([acq_polars[i],acq_types[i],precursors[i],mzs_in_filter])
                consolidated_filter_list.append([filters_list[i]])
                consolidated_idx_list.append([i])

        # Make a list where similar filters are grouped together to be used together
        mzs_per_filter_grp = []
        mzs_per_filter_grp_lb = []
        mzs_per_filter_grp_ub = []
        mz_idxs_per_filter_grp = []
        scans_per_filter_grp = np.empty((scans_per_filter.shape[0],0), dtype = int)
        for grp_idx, i in enumerate(consolidated_idx_list):
            mzs_per_filter_grp.append(mzsPerFilter[consolidated_idx_list[grp_idx][0]]) # find masses to pick in each group
            mzs_per_filter_grp_lb.append(mzsPerFilter_lb[consolidated_idx_list[grp_idx][0]])
            mzs_per_filter_grp_ub.append(mzsPerFilter_ub[consolidated_idx_list[grp_idx][0]])
            mz_idxs_per_filter_grp.append(mzIndicesPerFilter[consolidated_idx_list[grp_idx][0]])
            # get total number of scans for each group
            scans_for_grp_i = scans_per_filter[:,consolidated_idx_list[grp_idx]].sum(axis = 1)[:,None]
            scans_per_filter_grp = np.concatenate((scans_per_filter_grp,scans_for_grp_i), axis = 1)
        peak_counts_per_filter_grp = [len(i) for i in mzs_per_filter_grp]

        return consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, \
            scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list

    ## Works for D and mzML files, but is overwritten for raw files. 
    def reorder_pixels(self, pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs):
        """Reorders the pixels to match the order of the mass list."""
        # Initialize pixels with 1x1 images with values of 0.
        pixels_reordered = [np.zeros((1,1))]*(len(mass_list_idxs[0])+len(mass_list_idxs[1])+1)
        
        # For each filter group, check if MS1 or MS2    
        for i, filtr in enumerate(consolidated_filter_list):
            if filtr[0][2] in ['MS1', 1, '1']:
                pixels_reordered[0]=pixels[i][:,:,0]
                for j in range(pixels[i].shape[-1]-1):
                    pixels_reordered[mass_list_idxs[0][mz_idxs_per_filter_grp[i][j]]+1]=pixels[i][:,:,j+1]
            else:
                for j in range(pixels[i].shape[-1]-1):
                    pixels_reordered[mass_list_idxs[1][mz_idxs_per_filter_grp[i][j]]+1]=pixels[i][:,:,j+1]

        return pixels_reordered

    def pixels_list_to_array(self, pixels, all_TimeStamps_aligned):
        """Converts a list of pixels to a numpy array. Only to be used when all images are the same size."""
        for i, line in enumerate(pixels):
            if line.shape[1]==1 and all(line == 0):
                pixels[i]=np.zeros((len(self.line_list), len(all_TimeStamps_aligned[0])))
        return np.array(pixels)


    # =============================================================
    # Saving and loading output data
    # =============================================================

    def save_pixels(self, metadata=None, pixels=None, MSI_data_output=None, file_format = None, ask_confirmation = True):
        """
        Saves the pixels and metadata to a file in the specified format.
        The file format can be .npy, .npz, or .csv.
        .npy and .csv are used for saving images of the same size, whereas .npz is used for saving images of different sizes.
        If ask_confirmation is True, the user will be prompted to confirm overwriting existing files, otherwise it will overwrite them without asking.
        If an error occurs here, it will just be a warning and the program will continue.
        """
        save = True
        while save is True:
            if metadata is None:
                metadata = self.metadata
            if metadata is None:
                raise Exception("No metadata to save. Please run the get_image_data method first.")
            if pixels is None:
                pixels = self.pixels
            if pixels is None:
                raise Exception("No data to save. Please run the get_image_data method first.")
            if MSI_data_output is None:
                MSI_data_output = self.output_file_loc

            # decide on appropriate file extension
            if type(pixels) == type(np.zeros(0)):
                file_extension = ".npy"
            elif type(pixels) == list:
                file_extension = ".npz"

            if (self.normalize_img_sizes and file_extension in ["npz",".npz"]) or (file_format in ['csv','.csv'] and type(pixels) == list):
                pixels = self.resize_images_to_same_size(pixels)
            if file_format in ['csv','.csv']:
                file_extension = ".csv"
            elif file_format in ['npy','.npy']:
                file_extension = ".npy"

            # determine directory and file name based on the given path
            try:
                if MSI_data_output.split('.')[-1] in ['npy', 'npz', 'csv']:
                    # assume the given path is a filename
                    MSI_data_output_folder = os.path.split(MSI_data_output)[0]
                    MSI_data_output_filename = os.path.split(MSI_data_output)[-1]
                    MSI_data_output_filename = ".".join(MSI_data_output_filename.split(".")[:-1])+file_extension
                else:
                    # assume the given path is a folder
                    MSI_data_output_folder = MSI_data_output
                    MSI_data_output_filename = 'pixels'+file_extension
            except:
                warnings.warn("The given path is not a valid file or folder. The file will not be saved.")
                save = False
            
            try:
                # determine save paths
                pixels_path = os.path.join(MSI_data_output_folder,MSI_data_output_filename)

                metadata_filename = ".".join(MSI_data_output_filename.split('.')[:-1])+'_metadata.json'
                json_path = os.path.join(MSI_data_output_folder, metadata_filename)
            except:
                warnings.warn("The given path is not a valid file or folder. The file will not be saved.")
                save = False

            try:
                # make output folder
                if not os.path.exists(MSI_data_output_folder):
                    os.makedirs(MSI_data_output_folder)
            except:
                warnings.warn("The given path is not a valid file or folder. The file will not be saved.")
                save = False

            # check if files will be overwritten, and if so make a confirmation dialog box
            if ask_confirmation:
                if self.testing:
                    print("checking for existing files")
                overwrite_file = self.check_for_existing_files(json_path, pixels_path)
            else:
                overwrite_file = True
            
            if overwrite_file == False:
                save = False

            if overwrite_file:
                try:
                    # Save metadata
                    with open(json_path, 'w') as fp:
                        json.dump(metadata, fp, indent=4, cls=self.NpEncoder)
                    
                    # save pixels as .npy file if it is an array otherwise as an .npz archive
                    if file_extension == ".npy":
                        np.save(pixels_path, pixels, allow_pickle=True)

                    elif file_extension == ".npz":
                        np.savez(pixels_path, *pixels)

                    elif file_extension == ".csv":
                        # make array 2d dataframe and get x & y coordinates
                        indices = np.indices(pixels[0].shape).reshape(2,-1)
                        pixels_flattened = pixels.reshape((pixels.shape[0],-1))
                        column_headers = ['x','y']+[int(i) for i in range(pixels_flattened.shape[0])]
                        pixels_flattened = np.append(indices, pixels_flattened, axis = 0)
                        df = pd.DataFrame(pixels_flattened.T, columns=column_headers)
                        # save data
                        df.to_csv(pixels_path, float_format = '%.2f', index=False)

                    print(f'Saved data to {pixels_path} and {json_path}')
                except:
                    warnings.warn("The pixels file could not be saved.")
                    save = False
            break # prevents infinite loop if save stays true

    def check_for_existing_files(self, json_path, pixels_path):
        """Checks if the specified JSON and pixels files already exist."""
        try:
            existing_file_collector = []
            for i in [json_path, pixels_path]:
                if self.testing:
                    print("Checking for file {}".format(i))
                if os.path.exists(i):
                    existing_file_collector.append(i)
                    if self.testing:
                        print("File {} already exists.".format(i))
            if existing_file_collector:
                if self.testing:
                    print("The following files already exist:\n{}. Confirming whether they should be overwritten".format(existing_file_collector))
                overwrite_file = self.confirm_overwrite_file(existing_file_collector)
            else:
                if self.testing:
                    print("No existing files found.")
                overwrite_file = True
            return overwrite_file
        except:
            if self.testing:
                print("There was a failure in checking for existing files. Files will  not be saved.")
            return False
    
    def confirm_overwrite_file(self, file_list):
        """Prompts the user to confirm overwriting existing files."""
        if self.testing:
            print("opening dialogue")
        gc = self.get_confirmation_dialogue(file_list)
        if gc.response == "N":
            if self.testing:
                print("User chose not to overwrite files.")
            warnings.warn("Saving was cancelled to avoid overwriting previous files.")
            return False
        else:
            if self.testing:
                print("User chose to overwrite files.")
            return True

    def get_default_load_path(self):
        """
        Returns the default path for loading pixel data.
        Searches the working directory for a file named 'pixels.npz', 'pixels.npy', or 'pixels.csv'.
        """
        load_paths = []
        if self.output_file_loc.split('.')[-1].lower() in ["npy", "npz", "csv"]:
            load_paths.append(self.output_file_loc)
        else:
            load_paths.append(os.path.join(self.output_file_loc,'pixels.npy'))
            load_paths.append(os.path.join(self.output_file_loc,'pixels.npz'))
            load_paths.append(os.path.join(self.output_file_loc,'pixels.csv'))

        for load_path in load_paths:
            if os.path.exists(load_path):
                return load_path

        raise Exception('The file to load could not be found.')

    # TODO: Implement a version check for the metadata file to ensure compatibility with outputs from older MSIGen versions.
    def load_pixels(self, path=None):
        """
        Loads pixel data from the specified file without initializing the class beforehand.
        These files can be in the .npz, .npy, or .csv format and must have a corresponding metadata file in .json format.
        If path is None, it uses the default load path.
        """
        # check for default path
        if not path:
            path = self.get_default_load_path()
        
        # check if given path exists
        if not os.path.exists(path):
            raise FileNotFoundError('Data path given does not exist')

        # If path is a directory, search for a pixels.npy or npz file within it
        if os.path.isdir(path):
            
            MSI_data_path = None
            filenames_to_check = ['pixels.csv','pixels.npz','pixels.npy']
            
            for i in filenames_to_check:
                load_path = os.path.join(path, i)
                if os.path.exists(load_path):
                    MSI_data_path = load_path

            if MSI_data_path == None:
                raise FileNotFoundError('Data path given does not contain a pixels.npz, pixels.npy, or pixels.csv file')
        
        # use path if it is a file
        else: MSI_data_path = path

        # load pixels
        if MSI_data_path.endswith('.npz'):
            with np.load(MSI_data_path) as file:
                pixels = [file['arr_'+str(i)] for i in range(len(file.files))]

        elif MSI_data_path.endswith('.npy'):
            pixels = np.load(MSI_data_path, allow_pickle=True)

        elif MSI_data_path.endswith('.csv'):
            df = pd.read_csv(MSI_data_path)
            idxs = df.iloc[:,:2].astype(int).values
            pixels = df.iloc[:,2:].values.T
            pixels = pixels.reshape((pixels.shape[0],)+tuple(idxs.max(axis = 0)+1))

        else:
            raise ValueError('The file to load must be a .npy, .csv, or .npz file.')

        # load metadata
        metadata_path = '.'.join(MSI_data_path.split('.')[:-1])+'_metadata.json'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError('Associate metadata .json file was not found.')
        with open(metadata_path) as file:
            metadata = json.load(file)
        
        print(f'Loaded pixels from {MSI_data_path}\n\
        and metadata from {metadata_path}')
        
        return pixels, metadata
    
    def resize_images_to_same_size(self, pixels):
        """Resizes all images in the pixels list to the same size."""
        # if pixels is not an array, resize any images that are smaller than the largest image then save as an array
        if type(pixels) == list:
            
            # get shapes of each image
            sizes = [i.shape for i in pixels]
            max_size = np.max(sizes, axis = 0)

            # assign resized images to array
            new_pixels = np.zeros((np.append(len(pixels),max_size)), dtype = float)
            for i, size in enumerate(sizes):
                if np.all(size == max_size):
                    new_pixels[i] = pixels[i]
                else:
                    new_pixels[i] = resize(pixels[i], max_size, order=0)
            pixels = new_pixels
        
        return pixels

    # ===================================================
    # 1.3 Metadata related functions
    # ===================================================
    def get_basic_instrument_metadata(self, *args, **kwargs):
        raise NotImplementedError("This class does not support this method. Check that the proper subclass is being used.")

    def make_metadata_dict(self):
        """Creates a metadata dictionary containing information about the mass list, tolerances, and other parameters."""
        # unpack_variables
        MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs = self.mass_list
        mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance = self.tolerances
        mass_tolerance_MS1_units, mass_tolerance_prec_units, mass_tolerance_frag_units, mobility_tolerance_units = self.tolerance_units
        timestamp = os.stat(self.line_list[0]).st_mtime
        source_file_creation_date = datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d, %H:%M:%S")

        # get a mass list in the same order as pixels and save it to the metadata dict
        final_mass_list = [['TIC']]+[None]*(len(mass_list_idxs[0])+len(mass_list_idxs[1]))
        for counter, idx in enumerate(mass_list_idxs[0]):
            final_mass_list[idx+1]=[MS1_list[counter], MS1_mob_list[counter], MS1_polarity_list[counter]]
        for counter, idx in enumerate(mass_list_idxs[1]):
            final_mass_list[idx+1]=[prec_list[counter], frag_list[counter], MS2_mob_list[counter], MS2_polarity_list[counter]]
        
        # construct dictionary
        metadata = {"line_list":self.line_list,
                    "MS1_mass_list":MS1_list.tolist(),
                    "MS1_mobility_list":MS1_mob_list.tolist(),
                    "MS1_polarity_list":MS1_polarity_list.tolist(),
                    "precursor_mass_list":prec_list.tolist(),
                    "fragment_mass_list":frag_list.tolist(),
                    "MS2_mobility_list":MS2_mob_list.tolist(),
                    "MS2_polarity_list":MS2_polarity_list.tolist(),
                    "mass_list_idxs":mass_list_idxs,
                    "final_mass_list":final_mass_list,
                    "MS1_mass_tolerance":mass_tolerance_MS1,
                    "precursor_mass_tolerance":mass_tolerance_prec,
                    "fragment_mass_tolerance":mass_tolerance_frag,
                    "tolerance_mobility":mobility_tolerance,
                    "MS1_mass_tolerance_units":mass_tolerance_MS1_units,
                    "precursor_mass_tolerance_units":mass_tolerance_prec_units,
                    "fragment_mass_tolerance_units":mass_tolerance_frag_units,
                    "tolerance_mobility_units":mobility_tolerance_units,
                    "is_MS2":self.is_MS2,
                    "is_mobility":self.is_mobility,
                    "source_file_creation_date":source_file_creation_date,
                    "image_dimensions": [self.h, self.w],
                    "image_dimensions_units": self.hw_units,
                    "normalized_output_sizes": self.normalize_img_sizes,
                    "output_file_location": self.output_file_loc,
                    }
        
        return metadata

    def get_attr_values(self, metadata, source, attr_list, save_names = None, metadata_dicts = None):
        """Gets the values of the specified attributes from the metadata dictionary."""
        for i, attr in enumerate(attr_list):
            value = getattr(source, attr)
            if callable(value):
                value = str(value())
            if metadata_dicts:
                if metadata_dicts[i]:
                    try:
                        value = metadata_dicts[i][value]
                    except:
                        pass
            if save_names:
                if save_names[i]:
                    attr = save_names[i]
            metadata[attr]= str(value)
        return metadata