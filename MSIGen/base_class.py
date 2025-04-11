# =================================
# imports
# =================================

# General functions
import os, sys
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
# TODO Make the code class based with subclasses in D.py, mzml.py, and raw.py.

# non-class specific functions:
def _display_df(df):  
    pd.set_option("display.max_rows", None)
    iPydisplay(HTML(("<div style='max-height: 400px'>" + df.to_html() + "</div>")))
    pd.set_option("display.max_rows", 30)

# Used to suppress useless and repeated warnings
class HiddenPrints:
    '''Allows code to be run without displaying messages.'''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Used to save numpy data to Json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class get_confirmation():
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
        """ get the contents of the Entry and exit
        """
        self.response=value
        self.master.destroy()

class MSIGen_base(object):
    """Class for the MSIGen package. This class is used to generate MSI images from
    line scans in  raw, mzML, or .d files."""
    def __init__(self, example_file, mass_list_dir, tol_MS1=10, tol_MS1_u='ppm', tol_prec=1, tol_prec_u='mz', tol_frag=10, tol_frag_u='ppm', \
                 tol_mob=0.1, tol_mob_u='μs', h=10, w=10, hw_units='mm', is_MS2 = False, is_mobility=False, normalize_img_sizes = True, \
                 pixels_per_line = "mean", output_file_loc = None, in_jupyter = True, testing = False, gui = False):
        
        self.example_file = example_file
        self.file_ext = self.get_file_extension(example_file)
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
        self.get_metadata_and_params()
        if self.in_jupyter and (not self.testing):
            try: self.display_mass_list()
            except: pass

    @staticmethod
    def get_file_extension(example_file):
        """Static method to determine the proper subclass to use"""
        if type(example_file) in [list, tuple]:
            example_file = example_file[0]
        
        return os.path.splitext(example_file)[-1].lower()

    def get_metadata_and_params(self, **kwargs):
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
            if file.endswith(name_post):
                if file.startswith(name_body):
                    raw_files.append(file)

        return raw_files

    def sort_raw_files(self, raw_files, name_body = None, name_post = None):
        if name_body is None:
            name_body = self.name_body
        if name_post is None:
            name_post = self.name_post
        setattr(self, 'name_body', name_body)
        setattr(self, 'name_post', name_post)

        # array of the number of each line file name
        line_nums = np.array([int(file.replace(name_body,'').replace(name_post,'')) for file in raw_files])

        # sort the line by their numbers in ascending order
        sorted_raw_files = []
        for i in np.argsort(line_nums):
            sorted_raw_files.append(raw_files[i])
        return sorted_raw_files
    
    # =======================================================
    # Mass list functions
    # =======================================================
    def get_mass_list(self, mass_list_dir = None, header = 0, sheet_name = 0):
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
        col_types_filtered = []
        col_idxs = []

        for i, col_type in enumerate(col_types):
            if col_type not in col_types_filtered and not None:
                col_types_filtered.append(col_type)
                col_idxs.append(i)

        return col_types_filtered, col_idxs
    
    def get_mass_mobility_lists(self, raw_mass_list = None, col_type = None, col_idxs = None):
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
        '''Displays your mass/mobility list as a dataframe'''
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
        '''Determines the upper and lower bounds for selection window'''
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
        '''Determines the upper and lower bounds for each mass or mobility window'''
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
        premissible_keys = ['verbose', 'in_jupyter', 'testing', 'gui', 'results', 'pixels_per_line', 'tkinter_widgets']
        for key, value in kwargs.items():
            if key not in premissible_keys:
                raise Exception(f"Invalid keyword argument: {key}")
            setattr(self, key, value)
        
        self.lower_lims, self.upper_lims = self.get_all_ms_and_mobility_windows()

        self.metadata = self.make_metadata_dict()

        self.metadata, self.pixels = self.load_files(metadata = self.metadata)

        if self.gui:
            self.results["metadata"] = self.metadata
            self.results["pixels"] = self.pixels
        
        self.save_pixels(self.metadata, self.pixels)
        

        return self.metadata, self.pixels

    def load_files(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")

    def get_scan_without_zeros(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms1_no_mob(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms2_no_mob(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms1_mob(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def ms2_mob(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    def check_dim(self, *args, **kwargs):
        raise NotImplementedError("This class does not support loading files. Check that the proper subclass is being used.")
    
    # ============================================
    # GUI progressbar methods
    # ============================================
    def progressbar_start_preprocessing(self):
        if self.gui:
            self.tkinter_widgets[1]['text']="Preprocessing data"
            self.tkinter_widgets[1].update()

    def progressbar_start_extraction(self):
        if self.gui:
            self.tkinter_widgets[1]['text']="Extracting data"
            self.tkinter_widgets[1].update()

    def progressbar_update_progress(self, num_spe, i, j):
        # Update gui variables            
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
        if self.numba_present:
            idxs_to_sum = self.vectorized_sorted_slice_njit(mz, lb, ub)
            pixel = self.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
        else:
            idxs_to_sum = self.vectorized_sorted_slice(mz, lb, ub) # Slower
            pixel = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)
        return pixel

    @staticmethod
    def sorted_slice(a,l,r):
        '''
        Slices numpy array 'a' based on a given lower and upper bound.
        Array must be sorted for this to be used.
        '''
        start = np.searchsorted(a, l, 'left')
        end = np.searchsorted(a, r, 'right')
        # print(np.arange(start,end))
        return np.arange(start,end)

    @staticmethod
    def vectorized_sorted_slice(a,l,r):
        '''
        Slices numpy array 'a' based on given vectors of lower and upper bounds.
        Array must be sorted for this to be used.
        '''
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
        '''
        Gets indices of numpy array 'mz' within given vectors of lower and upper bounds.
        '''
        mass_idxs, int_idxs = np.where((mz[None,:]>lbs[:,None])&(mz[None,:]<ubs[:,None]))
        ls = [int_idxs[mass_idxs == i].tolist() for i in range(len(lbs))]
        max_num_idxs = max([len(i) for i in ls])
        if not max_num_idxs:
            max_num_idxs = 1
        ls = [i+[-1]*(max_num_idxs-len(i)) for i in ls]
        return np.array(ls)

    @staticmethod
    def vectorized_unsorted_slice_mob(mz,mob,lbs,ubs,mob_lbs,mob_ubs):
        '''
        Gets indices of numpy arrays 'mz' and 'mob' within given vectors of lower and upper bounds.
        '''
        mass_idxs, int_idxs = np.where((mz[None,:]>lbs[:,None])&(mz[None,:]<ubs[:,None])&(mob[None,:]>mob_lbs[:,None])&(mob[None,:]<mob_ubs[:,None]))
        ls = [int_idxs[mass_idxs == i].tolist() for i in range(len(lbs))]
        max_num_idxs = max([len(i) for i in ls])
        if not max_num_idxs:
            max_num_idxs = 1
        ls = [i+[-1]*(max_num_idxs-len(i)) for i in ls]
        return np.array(ls)

    # Since numba is an optional package and these imports will fail if njit is not imported from numba, these are only defined if it's present
    if "numba" in sys.modules:
        # Potential improvement to vectorized_sorted_slice using numba
        @staticmethod
        @njit
        def vectorized_sorted_slice_njit(a,l,r):
            '''
            Slices numpy array 'a' based on a given lower and upper bound.
            Array must be sorted for this to be used.
            '''
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
        def assign_values_to_pixel_njit(intensities, idxs_to_sum):
            return np.sum(np.take(intensities, idxs_to_sum), axis = 1)
    
    @staticmethod
    def flatten_list(l):
        return [item for sublist in l for item in sublist]

    # ============================================
    # MS1 specific data processing functions
    # ============================================

    def ms1_interp(self, pixels, rts = None, mass_list = None, pixels_per_line = None):
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
            rts_aligned = np.linspace(0, 1, int(np.mean([len(rts) for rts in rts_normed])))
        elif self.pixels_per_line == "min":
            rts_aligned = np.linspace(0, 1, int(np.mean([len(rts) for rts in rts_normed])))
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
        if pixels_per_line is not None:
            self.pixels_per_line = pixels_per_line
        if normalize_img_sizes is not None:
            self.normalize_img_sizes = normalize_img_sizes

        # Normalize timestamps to align each line in case one line took longer or started later.
        all_TimeStamps_normed  = self.normalize_ms2_timestamps(all_TimeStamps, acq_times)

        # Deterime how many pixels to use for each group of transitions and get evenly spaced times to sample at
        num_spe_per_group_aligned = self.get_num_spe_per_group_aligned(scans_per_filter_grp)
        print(num_spe_per_group_aligned)
        all_TimeStamps_aligned = [np.linspace(0,1,i) for i in num_spe_per_group_aligned]

        # make the final output of shape (lines, pixels_per_line, num_transitions+1)
        pixels = [np.zeros((len(self.line_list), num_spe_per_group_aligned[i], len(mzs)+1)) for (i, mzs) in enumerate(mzs_per_filter_grp)]

        # go through the extracted data and place them into pixels_final. list by group idx with shapes (# of lines, # of Pixels per line, m/z)
        for i, pixels_meta in enumerate(pixels_metas):
            for j, pixels_meta_grp in enumerate(pixels_meta):
                points = (all_TimeStamps_normed[i][j], np.arange(pixels_meta_grp.shape[1]))
                sampling_points = np.array(np.meshgrid(*(all_TimeStamps_aligned[j], np.arange(pixels_meta_grp.shape[1])), indexing = 'ij')).transpose(1,2,0)
                pixels[j][i] = interpn(points, pixels_meta_grp, sampling_points, method = 'nearest', bounds_error = False, fill_value=None)
        return pixels, all_TimeStamps_aligned
    
    def normalize_ms2_timestamps(self, all_TimeStamps, acq_times):
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

    def get_PeakCountsPerFilter(self, filters_info):
        '''
        Gets information about the peaks present in each filter.
        '''
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

        PeakCountsPerFilter = np.zeros((filter_list.shape)).astype(int)
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
                        if (acq_type in ['Full ms', 'MS1']) \
                            and (mz >= float(mz_range[0])) & (mz <= float(mz_range[1])) \
                            and (list_polarity in [0., polarity]):
                            
                            PeakCountsPerFilter[j] += 1
                            mzsPerFilter[j].append(mz)
                            mzsPerFilter_lb[j].append(MS1_lb[i])
                            mzsPerFilter_ub[j].append(MS1_ub[i])
                            mzIndicesPerFilter[j].append(i)
                        
                    else:
                        mob_range = mob_ranges[j]
                        mob = MS1_mob_list[i]
                        if (acq_type in ['Full ms', 'MS1']) \
                            and (mz >= float(mz_range[0])) & (mz <= float(mz_range[1])) \
                            and (mob >= float(mob_range[0])) & (mob <= float(mob_range[1])) \
                            and (list_polarity in [0., polarity]):

                            PeakCountsPerFilter[j] += 1
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
                        if (acq_type in ['Full ms2', 'MS2', 'MRM', 'diaPASEF', 'ddaPASEF']) \
                            and ((prec >= float(prec_lb[i])) & (prec <= float(prec_ub[i]))) \
                            and (list_frag >= float(frag_range[0])) & (list_frag <= float(frag_range[1])) \
                            and (list_polarity in [0., polarity]):
                                
                            PeakCountsPerFilter[j] += 1
                            mzsPerFilter[j].append(list_frag)
                            mzsPerFilter_lb[j].append(frag_lb[i])
                            mzsPerFilter_ub[j].append(frag_ub[i])

                            mzIndicesPerFilter[j].append(i)
                    
                    else:
                        mob_range = mob_ranges[j]
                        if (acq_type in ['Full ms2', 'MS2', 'MRM', 'diaPASEF', 'ddaPASEF']) \
                            and ((prec >= float(prec_lb[i])) & (prec <= float(prec_ub[i]))) \
                            and (list_frag >= float(frag_range[0])) & (list_frag <= float(frag_range[1])) \
                            and (list_mob >= float(mob_range[0])) & (list_mob <= float(mob_range[1])) \
                            and (list_polarity in [0., polarity]):

                            PeakCountsPerFilter[j] += 1
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
            return PeakCountsPerFilter, mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter
        else:
            return PeakCountsPerFilter, mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mobsPerFilter_lb, mobsPerFilter_ub, mzIndicesPerFilter

    def consolidate_filter_list(self, filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter):
        '''Deals with the case where ms2 filters do not have matching mass ranges'''

        # unpack filters_info
        filters_list, acq_polars, acq_types, precursors = filters_info[:4]

        # Make a boolean array where the filters match:
        # precursors, polarity, mode, and have the same fragments in the mass range
        # bool_arr = np.eye(len(mzsPerFilter), dtype = bool)
        groups = []
        consolidated_filter_list = []
        consolidated_idx_list = []

        for i, mzs_in_filter in enumerate(mzIndicesPerFilter):
            used = False
            for idx, group in enumerate(groups):
                if [acq_polars[i],acq_types[i],precursors[i],mzs_in_filter] == group:
                    used = True
                    consolidated_filter_list[idx].append(filters_list[i])
                    consolidated_idx_list[idx].append(i)
                    break
            if used == False:
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
        for i, line in enumerate(pixels):
            if line.shape[1]==1 and all(line == 0):
                pixels[i]=np.zeros((len(self.line_list), len(all_TimeStamps_aligned[0])))
        return np.array(pixels)


    # =============================================================
    # Saving and loading output data
    # =============================================================

    def save_pixels(self, metadata=None, pixels=None, MSI_data_output=None, file_format = None, ask_confirmation = True):
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

            if (file_format in ['csv','.csv','npy','.npy']) and (file_extension == ".npz"):
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

    def check_for_existing_files(self, json_path, pixels_path):
        try:
            existing_file_collector = []
            for i in [json_path, pixels_path]:
                if os.path.exists(i):
                    existing_file_collector.append(i)
            if existing_file_collector:
                overwrite_file = self.confirm_overwrite_file(existing_file_collector)
            else:
                overwrite_file = True
            return overwrite_file
        except:
            return False
    
    def confirm_overwrite_file(self, file_list):
        gc = self.get_confirmation_dialogue(file_list)
        if gc.response == "N":
            raise Warning("Saving was cancelled to avoid overwriting previous files.")
            return False
        else:
            return True

    def get_default_load_path(self):
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

    def load_pixels(self, path=None):

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

        # load metadata
        metadata_path = '.'.join(MSI_data_path.split('.')[:-1])+'_metadata.json'
        with open(metadata_path) as file:
            metadata = json.load(file)
        
        print(f'Loaded pixels from {MSI_data_path}\n\
        and metadata from {metadata_path}')
        
        return pixels, metadata
    
    def resize_images_to_same_size(pixels):
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