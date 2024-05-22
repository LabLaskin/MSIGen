# =================================
# imports
# =================================

# General functions
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interpn#, NearestNDInterpolator
from datetime import datetime
import json
from copy import deepcopy
from time import time
from operator import itemgetter

# GUI support
import tkinter as tk

try:
    from numba import njit
    numba_present = True
except:
    numba_present = False

# import MSIGen

try:
    jupyter_prints = True
    from IPython.display import display as iPydisplay
    from IPython.display import HTML
except:
    jupyter_prints = False

# TODO Make the code actually select based on the polarity.

# =====================================================
# 0.0 Collecting all metadata and parameters
# =====================================================

def get_metadata_and_params(example_file, mass_list_dir, \
    tol_MS1=10, tol_MS1_u='ppm', tol_prec=1, tol_prec_u='mz', tol_frag=10, tol_frag_u='ppm', tol_mob=0.1, tol_mob_u='μs',\
    h=10, w=10, hw_units='mm', is_MS2 = False, is_mobility=False, normalize_img_sizes = True, output_file_loc = None, in_jupyter = True, testing = False):

    tol_MS1, tol_MS1_u, tol_prec, tol_prec_u, tol_frag, tol_frag_u, tol_mob, tol_mob_u, h, w,\
        hw_units, is_MS2, is_mobility, normalize_img_sizes, output_file_loc, in_jupyter, testing \
    = get_metadata_and_params_handle_nones(tol_MS1, tol_MS1_u, tol_prec, tol_prec_u, tol_frag, tol_frag_u, tol_mob, tol_mob_u,\
        h, w, hw_units, is_MS2, is_mobility, normalize_img_sizes, output_file_loc, in_jupyter, testing)
    # list of file names for each line scan in the experiment in increasing order of increasing line number
    if type(example_file) in [list, tuple]:
        line_list = example_file
    else:
        line_list = get_line_list(example_file, display = False)
    # All mass, mobility, and polarity lists
    MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs = get_mass_list(mass_list_dir, header = 0, sheet_name = 0)
    mass_lists = MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs
    # all mass and mobility tolerances
    tolerances = tol_MS1, tol_prec, tol_frag, tol_mob
    tolerance_units = tol_MS1_u, tol_prec_u, tol_frag_u, tol_mob_u
    # integer value to determine if it is ms2 or mobility experiment for shorter code
    # {0 = MS1 & no mobility, 1 = MS2 & no mobility, 2 = MS1 & mobility, 3 = MS2 & mobility}
    experiment_type = int(is_MS2)+(2*int(is_mobility))

    # make metadata dict
    metadata = make_metadata_dict(line_list, mass_lists, tolerances, tolerance_units, experiment_type, h, w, hw_units, normalize_img_sizes, output_file_loc)

    if in_jupyter and (not testing):
        try: display_mass_list(metadata)
        except: pass

    return metadata

def get_metadata_and_params_handle_nones(tol_MS1=10, tol_MS1_u='ppm', tol_prec=1, tol_prec_u='mz', tol_frag=10, tol_frag_u='ppm', tol_mob=0.1, tol_mob_u='μs',\
    h=10, w=10, hw_units='mm', is_MS2 = False, is_mobility=False, normalize_img_sizes = True, output_file_loc = None, in_jupyter = True, testing = False):
    variables = [tol_MS1, tol_MS1_u, tol_prec, tol_prec_u, tol_frag, tol_frag_u, tol_mob, tol_mob_u, h, w, hw_units, is_MS2, is_mobility, normalize_img_sizes, output_file_loc, in_jupyter, testing]
    default = [10, 'ppm', 1, 'mz', 10, 'ppm', 0.1, 'μs', 10, 10, 'mm', False, False, True, None, False, False]
    for i, var in enumerate(variables):
        if var is None:
            var = default[i]
    
    return variables

def get_masslists_from_metadata(metadata):
    req_keys = ['MS1_mass_list', 'MS1_mobility_list', 'MS1_polarity_list', 'precursor_mass_list', 'fragment_mass_list', 'MS2_mobility_list', 'MS2_polarity_list', 'mass_list_idxs']
    MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs = itemgetter(*req_keys)(metadata)
    return [np.array(MS1_list), np.array(MS1_mob_list), np.array(MS1_polarity_list), np.array(prec_list), np.array(frag_list), np.array(MS2_mob_list), np.array(MS2_polarity_list), mass_list_idxs]

def get_tolerances_from_metadata(metadata):
    req_keys = ['MS1_mass_tolerance', 'precursor_mass_tolerance', 'fragment_mass_tolerance', 'tolerance_mobility']
    tol_MS1, tol_prec, tol_frag, tol_mob = itemgetter(*req_keys)(metadata)
    return [np.array(tol_MS1), np.array(tol_prec), np.array(tol_frag), np.array(tol_mob)]

def get_tolerance_units_from_metadata(metadata):
    req_keys = ['MS1_mass_tolerance_units', 'precursor_mass_tolerance_units', 'fragment_mass_tolerance_units', 'tolerance_mobility_units']
    tol_MS1_u, tol_prec_u, tol_frag_u, tol_mob_u = itemgetter(*req_keys)(metadata)
    return [tol_MS1_u, tol_prec_u, tol_frag_u, tol_mob_u]

# =====================================================
# 1.0 Functions for obtaining line_list
# =====================================================

def segment_filename(FileName):
    # Get the file extension
    Name, NamePost = os.path.splitext(FileName)

    # determine the length of the line number
    iterator = 0
    for i in Name[::-1]:
        if i.isdigit():
            iterator+=1
        else:
            break
    
    if iterator<1:
        raise Exception('File names must end in a number that identifies the line.')
    
    # Get the part of the file name that does not contain the line number
    NameBody = Name[:-iterator]

    return NameBody, NamePost

def get_raw_files(NameBody, NamePost):
    # list all files in directory as absolute paths
    directory = os.path.split(NameBody)[0]
    files_in_dir = [os.path.join(directory,i) for i in os.listdir(directory)]
    
    # remove all files that do not fit the same naming scheme as given file
    raw_files = []
    for file in files_in_dir:
        if file.endswith(NamePost):
            if file.startswith(NameBody):
                raw_files.append(file)

    return raw_files

def sort_raw_files(NameBody, NamePost, raw_files):
    # array of the number of each line file name
    line_nums = np.array([int(file.replace(NameBody,'').replace(NamePost,'')) for file in raw_files])
    
    # sort the line by their numbers in ascending order
    sorted_raw_files = []
    for i in np.argsort(line_nums):
        sorted_raw_files.append(raw_files[i])
    return sorted_raw_files

def get_line_list(ExampleFile, display = False):
    NameBody, NamePost = segment_filename(ExampleFile)
    raw_files = get_raw_files(NameBody, NamePost)
    line_list = sort_raw_files(NameBody, NamePost, raw_files)

    if display:
        if jupyter_prints:
            iPydisplay(line_list)
        else:
            print(line_list)

    return line_list

# ===================================================
# 1.0 Functions for obtaining mass and mobility lists
# ===================================================

def column_type_check(mass_list):
    # define the header keywords to search for
    kwds = {'MS1_mz_kwds': ['mz','m/z','mass','masses', 'exactmass', 'exactmasses', 'exact_mass', 'exact_masses'],
    'precursor_kwds'     : ['parent','parents','precursor','precursors','prec'],
    'fragment_kwds'      : ['child','fragment','fragments','frag'],
    'mobility_kwds'      : ['mobility','mob','ccs', 'dt', 'drifttime', 'drifttime(ms)', 'collision cross section', 'dt(ms)'],
    'polarity_kwds'      : ['polarity', 'charge']}

    # initialize list for saving column types
    col_type=[]

    # check through each keyword for column type and append to list
    for i, col_name in enumerate(mass_list.columns):
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

def select_mass_list_cols_to_use(col_types):
    # initialize lists
    col_types_filtered = []
    col_idxs = []

    # each unique column type that is not none is recorded along with the index it first appears at.
    for i, col_type in enumerate(col_types):
        if col_type not in col_types_filtered and not None:
            col_types_filtered.append(col_type)
            col_idxs.append(i)

    return col_types_filtered, col_idxs

# make sure there are column headders
def column_header_check(mass_list):
    # Ensure the column headers arent data values
    if 'float' in str(type(mass_list.columns[0])):
        raise Exception("Ensure that the mass list sheet has column headers. \n\
        For example: 'precursor', 'fragment', 'mobility'")
    
    col_type = column_type_check(mass_list)

    col_type, col_idxs = select_mass_list_cols_to_use(col_type)

    # This combination is not compatible
    if set(['MS1', 'precursor', 'fragment']).issubset(col_type):
        raise Exception("There must be at most two columns containing m/z values, \n\
one for the precursor m/z and one for the fragment m/z.")
    
    # Ensure at least one column has m/z values
    if not len([i for i in ['MS1', 'precursor', 'fragment'] if i in col_type]):
        raise Exception("There must be at least one column with m/z values \n\
with one of the following a column headers: 'm/z', 'fragment', 'precursor'")

    # if there is MS1 and precursor or fragment columns together, convert MS1 to the correct corresponding column type
    if set(['MS1', 'fragment']).issubset(col_type):
        for i, column in enumerate(col_type):
            if column == 'MS1':
                col_type[i] = 'precursor'
                break
    elif set(['MS1', 'precursor']).issubset(col_type):
        for i, column in enumerate(col_type):
            if column == 'MS1':
                col_type[i] = 'fragment'
                break
    elif not set(['MS1']).issubset(col_type) and (set(['precursor']).issubset(col_type) ^ set(['fragment']).issubset(col_type)):
        for i, column in enumerate(col_type):
            if column in ['precursor', 'fragment']:
                col_type[i] = 'MS1'
                break

    return col_type, col_idxs

def get_mass_mobility_lists(mass_list, col_type, col_idxs):
    mass_list = mass_list.iloc[:,col_idxs]
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

    return MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs

def get_mass_list(mass_list_dir, header = 0, sheet_name = 0):
    # read excel-style files
    if mass_list_dir.split('.')[-1].lower() in ['xls','xlsx','xlsm','xlsb','odf','ods','odt']:
        mass_list = pd.read_excel(mass_list_dir, header = 0, sheet_name = 0)

    # read csv files 
    elif mass_list_dir.lower().endswith('csv'):
        mass_list = pd.read_csv(mass_list_dir, header = header)

    else:
        raise Exception("Transition/Mass list file must be one of the following formats: 'xls','xlsx','xlsm','xlsb','odf','ods','odt','csv' ")

    col_type, col_idxs = column_header_check(mass_list)

    mass_lists = get_mass_mobility_lists(mass_list, col_type, col_idxs)
    return mass_lists

def define_tolerance_units(MS1_units='ppm', prec_units='ppm', frag_units='ppm', mob_units='μs'):
    return MS1_units, prec_units, frag_units, mob_units

def get_mass_or_mobility_window(upper_lims, lower_lims, val_list, tol, unit):
    '''Determines the upper and lower bounds for selection window'''
    if unit.lower() == 'ppm':
        lower_lims.append(np.clip(np.array(val_list) * (1-(tol/1000000)), 0, None))
        upper_lims.append(np.array(val_list) * (1+(tol/1000000)))
    else:
        lower_lims.append(np.clip((np.array(val_list) - tol), 0, None))
        upper_lims.append(np.array(val_list) + tol)
    return upper_lims, lower_lims

def get_all_ms_and_mobility_windows(mass_lists, tolerances, tolerance_units):
    '''Determines the upper and lower bounds for each mass or mobility window'''
    # unpack variables
    MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs = mass_lists
    mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance = tolerances
    mass_tolerance_MS1_units, mass_tolerance_prec_units, mass_tolerance_frag_units, mobility_tolerance_units = tolerance_units
    upper_lims = []
    lower_lims = []

    # get upper and lower limits in order:
    # MS1, MS1_mob, MS1_polarity, precursor, fragment, MS2_mob, MS2_polarity
    lower_lim, upper_lim = get_mass_or_mobility_window(upper_lims, lower_lims, MS1_list, mass_tolerance_MS1, mass_tolerance_MS1_units)
    lower_lim, upper_lim = get_mass_or_mobility_window(upper_lims, lower_lims, MS1_mob_list, mobility_tolerance, mobility_tolerance_units)
    lower_lims.append(np.array(MS1_polarity_list))
    upper_lims.append(np.array(MS1_polarity_list))
    lower_lim, upper_lim = get_mass_or_mobility_window(upper_lims, lower_lims, prec_list, mass_tolerance_prec, mass_tolerance_prec_units)
    lower_lim, upper_lim = get_mass_or_mobility_window(upper_lims, lower_lims, frag_list, mass_tolerance_frag, mass_tolerance_frag_units)
    lower_lim, upper_lim = get_mass_or_mobility_window(upper_lims, lower_lims, MS2_mob_list, mobility_tolerance, mobility_tolerance_units)
    lower_lims.append(np.array(MS2_polarity_list))
    upper_lims.append(np.array(MS2_polarity_list))

    return lower_lims, upper_lims    

# ===================================================
# 1.2 General Utility functions
# ===================================================

def display_mass_list(metadata):
    '''Displays your mass/mobility list as a dataframe'''
    mass_list = deepcopy(metadata['final_mass_list'])
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
    _display_df(pd.DataFrame(a, columns = ["Precursor", "Fragment", "Mobility", "Polarity"], index = range(1, len(a)+1)))

def _display_df(df):  
    pd.set_option("display.max_rows", None)
    iPydisplay(HTML(("<div style='max-height: 400px'>" + df.to_html() + "</div>")))
    pd.set_option("display.max_rows", 30)


# ===================================================
# 1.3 Metadata related functions
# ===================================================
def make_metadata_dict(line_list, mass_lists, tolerances, tolerance_units, experiment_type, img_height, img_width, hw_dim, normalize_img_sizes, output_file_loc):
    # unpack_variables
    MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs = mass_lists
    mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance = tolerances
    mass_tolerance_MS1_units, mass_tolerance_prec_units, mass_tolerance_frag_units, mobility_tolerance_units = tolerance_units
    is_MS2 = bool(experiment_type%2)
    is_mobility = bool(experiment_type>=2)
    timestamp = os.stat(line_list[0]).st_mtime
    source_file_creation_date = datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d, %H:%M:%S")

    # get a mass list in the same order as pixels and save it to the metadata dict
    final_mass_list = [['TIC']]+[None]*(len(mass_list_idxs[0])+len(mass_list_idxs[1]))
    for counter, idx in enumerate(mass_list_idxs[0]):
        final_mass_list[idx+1]=[MS1_list[counter], MS1_mob_list[counter], MS1_polarity_list[counter]]
    for counter, idx in enumerate(mass_list_idxs[1]):
        final_mass_list[idx+1]=[prec_list[counter], frag_list[counter], MS2_mob_list[counter], MS2_polarity_list[counter]]
    
    # construct dictionary
    metadata = {"line_list":line_list,
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
                "is_MS2":is_MS2,
                "is_mobility":is_mobility,
                "source_file_creation_date":source_file_creation_date,
                "image_dimensions": [img_height, img_width],
                "image_dimensions_units": hw_dim,
                "normalized_output_sizes": normalize_img_sizes,
                "output_file_location": output_file_loc,
                }
    
    return metadata

def get_attr_values(metadata, source, attr_list, save_names = None, metadata_dicts = None):
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


# ============================================
# Main Data Extraction Workflow Functions
# ============================================
def get_image_data(metadata, verbose = False, in_jupyter = True, testing = False, gui = False, results = {}, tkinter_widgets = [None, None, None]):
    line_list = metadata['line_list']
    mass_lists = get_masslists_from_metadata(metadata)
    tolerances = get_tolerances_from_metadata(metadata)
    tolerance_units = get_tolerance_units_from_metadata(metadata)

    # Determines all the lower and upper limits for each mass and mobility window
    lower_lims, upper_lims = get_all_ms_and_mobility_windows(mass_lists, tolerances, tolerance_units)

    # integer value to determine if it is ms2 or mobility experiment for shorter code
    # {0 = MS1 & no mobility, 1 = MS2 & no mobility, 2 = MS1 & mobility, 3 = MS2 & mobility}
    experiment_type = int(metadata['is_MS2'])+(2*int(metadata['is_mobility']))

    normalize_img_sizes = metadata['normalized_output_sizes']
    
    metadata, pixels = load_files(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes, verbose=verbose, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)

    # allows you to get output from thread
    if gui:
        results['metadata'] = metadata
        results['pixels'] = pixels

    save_pixels(metadata, pixels, MSI_data_output=metadata['output_file_location'], gui = ((gui or in_jupyter) and (not testing))) 

    return metadata, pixels

def load_files(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, verbose = False, in_jupyter=True, testing=False, gui = False, tkinter_widgets = [None, None, None]):
    '''Initiates the acquisition of images from line scan files.
    Calls sub-functions depending on the file extension.'''
    if not gui:
        # starting time
        t_i = time()

    # call function depending on file extension
    if line_list[0].lower().endswith('.raw'):
        metadata, pixels = load_files_raw(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, verbose = verbose, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
    elif line_list[0].lower().endswith('.d'):
        metadata, pixels = load_files_d(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, verbose = verbose, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
    elif line_list[0].lower().endswith('.mzml'):
        metadata, pixels = load_files_mzml(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, verbose = verbose, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)

    if not gui:
        # total elapsed time
        t_tot = time()-t_i
        t_min = t_tot//60
        t_s = round(t_tot - (t_min*60), 2)

        if t_min:
            print(f'Time elapsed: {t_min} min {t_s} s')
        else:
            print(f'Time elapsed: {t_s} s')

    return metadata, pixels

def load_files_raw(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, verbose = False, in_jupyter = True, testing = False, gui = False, tkinter_widgets = [None, None, None]):    
    from MSIGen.raw import raw_ms1_no_mob, raw_ms2_no_mob
    if experiment_type == 0:
        if verbose:
            print('The file was identified as a Thermo .raw file containing only MS1 data')
        metadata, pixels = raw_ms1_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
        
    elif experiment_type == 1:
        if verbose:
            print('The file was identified as a Thermo .raw file containing MS2 data')
        metadata, pixels = raw_ms2_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
    
    else:
        raise NotImplementedError('Currently, .raw files can not be processed with mobility data.')

    return metadata, pixels

def load_files_d(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, verbose = False, in_jupyter = True, testing = False, gui = False, tkinter_widgets = [None, None, None]):    
    from MSIGen.D import determine_file_format, agilent_d_ms1_no_mob, agilent_d_ms2_no_mob, tsf_d_ms1_no_mob, tsf_d_ms2_no_mob, tdf_d_ms1_mob, tdf_d_ms2_mob

    # get the vendor format and whether it contains MS2 data
    file_format, MS_level = determine_file_format(line_list)

    if MS_level == 'MS1':
        if experiment_type in [1,3]:
            raise Warning("The file being used has MS2 scans but MS1 was selected as the experiment type.")
    elif MS_level == 'MS2':
        if experiment_type in [0,2]:
            raise Warning("The file being used has only MS1 scans but MS2 was selected as the experiment type.")

    if file_format == 'Agilent':
        if experiment_type == 0:
            if verbose:
                print('The file was identified as a Agilent .d file containing only MS1 data')
            metadata, pixels = agilent_d_ms1_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
        elif experiment_type == 1:
            if verbose:
                print('The file was identified as a Agilent .d file containing MS2 data')
            metadata, pixels = agilent_d_ms2_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
        else:
            raise NotImplementedError('Currently, Agilent data with ion mobility cannot be directly accessed in .d format with python.\n\
    Please convert the file to .mzML format with MSConvert and try again.')

    elif file_format == "bruker_tsf":
        if experiment_type == 0:
            if verbose:
                print('The file was identified as a Bruker tsf .d file containing only MS1 data')
            metadata, pixels = tsf_d_ms1_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
        elif experiment_type == 1:
            if verbose:
                print('The file was identified as a Bruker tsf .d file containing MS2 data')
            metadata, pixels = tsf_d_ms2_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
        else: raise NotImplementedError('Bruker .tsf data format does not store mobility data.')
    
    elif file_format == "bruker_tdf":
        if experiment_type == 2:
            if verbose:
                print('The file was identified as a Bruker tdf .d file containing MS1 and mobility data')
            metadata, pixels = tdf_d_ms1_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
        elif experiment_type == 3:
            if verbose:
                print('The file was identified as a Bruker tdf .d file containing MS2 and mobility data')
            metadata, pixels = tdf_d_ms2_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
    else:
        raise NotImplementedError('Currently, this data format has not been implemented.')

    return metadata, pixels

def load_files_mzml(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, verbose = False, in_jupyter = True, testing = False, gui = False, tkinter_widgets = [None, None, None]):    
    from MSIGen.mzml import mzml_ms1_no_mob, mzml_ms2_no_mob, mzml_ms1_mob, mzml_ms2_mob

    if experiment_type == 0:
        if verbose:
            print('The file was identified as an .mzml file containing only MS1 data')
        metadata, pixels = mzml_ms1_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
        
    elif experiment_type == 1:
        if verbose:
            print('The file was identified as an .mzml file containing MS2 data')
        metadata, pixels = mzml_ms2_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = normalize_img_sizes, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)
    
    elif experiment_type == 2:
        if verbose:
            print('The file was identified as an .mzml file containing MS1 and mobility data')
        metadata, pixels = mzml_ms1_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)

    else:
        if verbose:
            print('The file was identified as an .mzml file containing MS2 and mobility data')
        metadata, pixels = mzml_ms2_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = normalize_img_sizes, in_jupyter=in_jupyter, testing=testing, gui=gui, tkinter_widgets = tkinter_widgets)

    return metadata, pixels

# ============================================
# General Data Processing Functions
# ============================================

    # ============================================
    # Slicing sorted data for mass selection 
    # ============================================
def sorted_slice(a,l,r):
    '''
    Slices numpy array 'a' based on a given lower and upper bound.
    Array must be sorted for this to be used.
    '''
    start = np.searchsorted(a, l, 'left')
    end = np.searchsorted(a, r, 'right')
    # print(np.arange(start,end))
    return np.arange(start,end)

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

if numba_present == True:
    # Potential improvement to vectorized_sorted_slice using numba
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
    @njit
    def assign_values_to_pixel_njit(intensities, idxs_to_sum):
        return np.sum(np.take(intensities, idxs_to_sum), axis = 1)

def flatten_list(l):
    return [item for sublist in l for item in sublist]

# ============================================
# MS1 specific data processing functions
# ============================================

def ms1_interp(pixels, rts, mass_list, line_list):
    # normalize the retention times to be [0-1] and find evenly spaced times to resample at
    rts_normed = [(line_rts - line_rts.min())/(line_rts.max() - line_rts.min()) for line_rts in rts]
    rts_aligned = np.linspace(0, 1, int(np.mean([len(rts) for rts in rts_normed])))

    # Initialize pixels
    pixels_aligned = np.empty([len(line_list), len(rts_aligned), (mass_list.shape[0]+1)])

    # Interpolate each line with nearest neighbor to align number of pixels per line
    X = np.arange(pixels_aligned.shape[-1])
    for idx, line in enumerate(pixels):
        coords = (rts_normed[idx], X)
        line_pixels_aligned = interpn(coords, line, np.array(np.meshgrid(rts_aligned,X)).reshape(2, -1).transpose(1,0), method='nearest').reshape(X.size, rts_aligned.size)
        pixels_aligned[idx] = line_pixels_aligned.T
        
    # makes axes (m/z, h, w)
    pixels_aligned = np.moveaxis(pixels_aligned, -1, 0)
    
    return pixels_aligned

def ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, normalize_img_sizes, mzs_per_filter_grp, line_list):
    # Normalize timestamps to align each line in case one line took longer or started later.
    all_TimeStamps_normed  = normalize_ms2_timestamps(all_TimeStamps, acq_times)

    # Deterime how many pixels to use for each group of transitions and get evenly spaced times to sample at
    num_spe_per_group_aligned = get_num_spe_per_group_aligned(scans_per_filter_grp, normalize_img_sizes)
    all_TimeStamps_aligned = [np.linspace(0,1,i) for i in num_spe_per_group_aligned]

    # make the final output of shape (lines, pixels_per_line, num_transitions+1)
    pixels = [np.zeros((len(line_list), num_spe_per_group_aligned[i], len(mzs)+1)) for (i, mzs) in enumerate(mzs_per_filter_grp)]

    # go through the extracted data and place them into pixels_final. list by group idx with shapes (# of lines, # of Pixels per line, m/z)
    for i, pixels_meta in enumerate(pixels_metas):
        for j, pixels_meta_grp in enumerate(pixels_meta):
            points = (all_TimeStamps_normed[i][j], np.arange(pixels_meta_grp.shape[1]))
            sampling_points = np.array(np.meshgrid(*(all_TimeStamps_aligned[j], np.arange(pixels_meta_grp.shape[1])), indexing = 'ij')).transpose(1,2,0)
            pixels[j][i] = interpn(points, pixels_meta_grp, sampling_points, method = 'nearest', bounds_error = False, fill_value=None)
    return pixels, all_TimeStamps_aligned

# ============================================
# MS2 specific data processing functions
# ============================================

# Made unnecessary by combining this function with the sub functions.
# def check_dim(line_list, experiment_type = 0, ShowNumLineSpe=False):    
#     """Gets the times and other information about each scan to decide 
#     what peaks can be obtained from each scan.
#     Calls a sub-function depending on the file extension."""
#     # Process based on data type
#     if line_list[0].lower().endswith('.mzml'):

#         acq_times, all_filters_list = MSIGen.mzml.check_dim_mzml(line_list, experiment_type)
#     elif line_list[0].lower().endswith('.raw'):
#         acq_times, all_filters_list = MSIGen.raw.check_dim_raw(line_list, experiment_type)
#     elif line_list[0].lower().endswith('.d'):
#         acq_times, all_filters_list = MSIGen.D.check_dim_d(line_list, experiment_type)
    
#     num_spe_per_line = [len(i) for i in acq_times]
#     # show results
#     if ShowNumLineSpe:
#         print('\nline scan spectra summary\n# of lines is: {}\nmean # of spectra is: {}\nmin # of spectra is: {}\nmean start time is {}\nmean end time is: {}'.format(
#             len(num_spe_per_line), int(np.mean(num_spe_per_line)), int(np.min(num_spe_per_line)),np.mean([i[0] for i in acq_times]),np.mean([i[-1] for i in acq_times])))
#     return acq_times, all_filters_list

def get_PeakCountsPerFilter(filters_info, mass_lists, lower_lims, upper_lims):
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
    MS1_list, MS1_mob_list, MS1_polarity_list, prec_list, frag_list, MS2_mob_list, MS2_polarity_list, mass_list_idxs = mass_lists
    MS1_lb, MS1_mob_lb, _, prec_lb, frag_lb, ms2_mob_lb, _ = lower_lims
    MS1_ub, MS1_mob_ub, _, prec_ub, frag_ub, ms2_mob_ub, _ = upper_lims

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
                # print(acq_type in ['Full ms', 'MS1'])
                
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

def consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter):
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

    return consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list

def normalize_ms2_timestamps(all_TimeStamps, acq_times):
    all_TimeStamps_normed = []
    for i, line_timestamps in enumerate(all_TimeStamps):
        t_max = max(acq_times[i])
        t_min = min(acq_times[i])
        all_TimeStamps_normed.append([])
        for grp_timestamps in line_timestamps:
            all_TimeStamps_normed[i].append((grp_timestamps-t_min)/(t_max-t_min))
    return all_TimeStamps_normed

def get_num_spe_per_group_aligned(scans_per_filter_grp, normalize_img_sizes=True):
    num_spe_per_group_aligned = np.ceil(np.max(np.array(scans_per_filter_grp), axis = 0)).astype(int)
    if normalize_img_sizes == True:
        num_spe_per_group_aligned = np.full(num_spe_per_group_aligned.shape, num_spe_per_group_aligned.max(), dtype = int)
    return num_spe_per_group_aligned

def reorder_pixels(pixels, filters_grp_info, mz_idxs_per_filter, mass_list_idxs, line_list, filters_info = None):
    # get the scan type/level 
    if line_list[0].lower().endswith('.raw'):
        iterator = [] 
        for filter_grp in filters_grp_info:
            iterator.append(filters_info[2][np.where(filter_grp[0]==filters_info[0])])
    else:
        iterator = [filtr[0][2] for filtr in filters_grp_info]

    #put pixels into a list. If the window is MS1, its first mass will be assumed to be TIC.
    pixels_reordered = [np.zeros((1,1))]*(len(mass_list_idxs[0])+len(mass_list_idxs[1])+1)
    for i, acq_type in enumerate(iterator):
        if acq_type in ['MS1', 1, '1', 'Full ms']:
            pixels_reordered[0] = pixels[i][:,:,0]
            for j in range(pixels[i].shape[-1]-1):
                pixels_reordered[mass_list_idxs[0][mz_idxs_per_filter[i][j]]+1]=pixels[i][:,:,j+1]
        else:
            for j in range(pixels[i].shape[-1]-1):
                pixels_reordered[mass_list_idxs[1][mz_idxs_per_filter[i][j]]+1]=pixels[i][:,:,j+1]

    return pixels_reordered

def pixels_list_to_array(pixels, line_list, all_TimeStamps_aligned):
    for i, line in enumerate(pixels):
        if line.shape[1]==1 and all(line == 0):
            pixels[i]=np.zeros((len(line_list), len(all_TimeStamps_aligned[0])))
    return np.array(pixels)


# =============================================================
# Saving and loading output data
# =============================================================

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

def save_pixels(metadata, pixels, MSI_data_output=None, file_format = None, compress = False, gui = True):

    # decide on appropriate file extension
    if type(pixels) == type(np.zeros(0)):
        file_extension = ".npy"
    elif type(pixels) == list:
        file_extension = ".npz"

    if file_format in ['csv','.csv','npy','.npy']:
        if file_extension == ".npz":
            pixels = normalize_img_sizes(pixels)
    if file_format in ['csv','.csv']:
        file_extension = ".csv"
    elif file_format in ['npy','.npy']:
        file_extension = ".npy"

    if not MSI_data_output:
        try: 
            MSI_data_output = metadata['output_file_location']
            assert type(MSI_data_output) == str
        except:
            MSI_data_output = os.path.join(os.getcwd(),'pixels.npy')

    # determine directory and file name based on the given path
    if MSI_data_output.split('.')[-1] in ['npy', 'npz', 'csv']:
        # assume the given path is a filename
        MSI_data_output_folder = os.path.split(MSI_data_output)[0]
        MSI_data_output_filename = os.path.split(MSI_data_output)[-1]
        MSI_data_output_filename = ".".join(MSI_data_output_filename.split(".")[:-1])+file_extension
    else:
        # assume the given path is a folder
        MSI_data_output_folder = MSI_data_output
        MSI_data_output_filename = 'pixels'+file_extension

    # determine save paths
    pixels_path = os.path.join(MSI_data_output_folder,MSI_data_output_filename)

    metadata_filename = ".".join(MSI_data_output_filename.split('.')[:-1])+'_metadata.json'
    json_path = os.path.join(MSI_data_output_folder, metadata_filename)

    # make output folder
    if not os.path.exists(MSI_data_output_folder):
        os.makedirs(MSI_data_output_folder)

    # check if files will be overwritten, and if so make a confirmation dialog box
    if gui:
        overwrite_file = check_for_existing_files(json_path, pixels_path)
    else:
        overwrite_file = True

    if overwrite_file:

        # Save metadata
        with open(json_path, 'w') as fp:
            json.dump(metadata, fp, indent=4, cls=NpEncoder)
        
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

def check_for_existing_files(json_path, pixels_path):
    existing_file_collector = []
    for i in [json_path, pixels_path]:
        if os.path.exists(i):
            existing_file_collector.append(i)
    if existing_file_collector:
        overwrite_file = confirm_overwrite_file(existing_file_collector)
    else:
        overwrite_file = True
    return overwrite_file
    
def confirm_overwrite_file(file_list):
    gc = get_confirmation(file_list)
    if gc.response == "N":
        raise Warning("Saving was cancelled to avoid overwriting previous files.")
        return False
    else:
        return True

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

def get_default_load_path():
    load_paths = []
    load_paths.append(os.path.join(os.getcwd(),'pixels.npy'))
    load_paths.append(os.path.join(os.getcwd(),'pixels.npz'))
    load_paths.append(os.path.join(os.getcwd(),'pixels.csv'))

    for load_path in load_paths:
        if os.path.exists(load_path):
            return load_path

    raise Exception('The file to load could not be found.')

def load_pixels(path=None):

    # check for default path
    if not path:
        path = get_default_load_path()
    
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
    
def normalize_img_sizes(pixels):
    from skimage.transform import resize

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
                
