from MSIGen import msigen

# mzML access
import pymzml

import os, sys
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interpn
from time import time


# ======================================================================
# MS1 without mobility 
# ======================================================================

def mzml_ms1_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    # get mass windows
    MS1_list, _, MS1_polarity_list, prec_list, frag_list, _, MS2_polarity_list, mass_list_idxs = mass_lists
    lb, _, _, _, _, _, _ = lower_lims
    ub, _, _, _, _, _, _ = upper_lims

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    # initiate accumulator
    pixels = []
    rts = []

    for i, file_dir in tqdm(enumerate(line_list), total = len(line_list), desc='Progress through lines', disable = (testing or gui)): 
        
        with pymzml.run.Reader(file_dir, obo_version = '4.1.9') as reader:

            # if i == 0:
            #     metadata = get_basic_instrument_metadata_mzml_no_mob(line_list, data, metadata)

            # grab headers for all scans
            num_spe = reader.get_spectrum_count()
            assert num_spe>0, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)

            line_rts = np.zeros(num_spe)
            line_pixels = np.zeros((num_spe, len(lb)+1))
            
            for j, spectrum in enumerate(reader):
                # Update gui variables            
                if gui:
                    tkinter_widgets[0]['value']=(100*i/len(line_list))+((100/len(line_list))*(j/num_spe))
                    tkinter_widgets[0].update()
                    tkinter_widgets[2]['text']=f'line {i+1}/{len(line_list)}, spectrum {j+1}/{num_spe}'
                    tkinter_widgets[2].update()

                # save scan time and TICs
                line_rts[j] = spectrum.scan_time[0]
                TIC = spectrum.TIC
                line_pixels[j,0] = TIC

                # get mz and intensity values
                mz = spectrum.mz
                intensity_points = np.append(spectrum.i,0)

                if msigen.numba_present:
                    idxs_to_sum = msigen.vectorized_sorted_slice_njit(mz, lb, ub)
                    pixel = msigen.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                    line_pixels[j,1:] = pixel
                else:
                    idxs_to_sum = msigen.vectorized_sorted_slice(mz, lb, ub) # Slower
                    line_pixels[j,1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)


        pixels.append(line_pixels)
        rts.append(line_rts)

    metadata['average_start_time'] = np.mean([i[0] for i in rts])
    metadata['average_end_time'] = np.mean([i[-1] for i in rts])

    pixels_aligned = msigen.ms1_interp(pixels, rts, MS1_list, line_list)
    
    return metadata, pixels_aligned

class HiddenPrints:
    '''Allows code to be run without displaying messages.'''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ======================================================================
# MS1 with mobility 
# ======================================================================

# MAKE SURE MOBILITY DATA ARE COMBINED WHEN USING MSCONVERT
def mzml_ms1_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    MS1_list, MS1_mob_list, MS1_polarity_list, _, _, _, _, mass_list_idxs = mass_lists

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    pixels_meta = []
    acq_times = []

    # Get the mass ranges and determine the minimum and max mass and mobility values that may be used.
    mz_lb, mz_ub = lower_lims[0], upper_lims[0]
    mob_lb, mob_ub = lower_lims[1], upper_lims[1]
    mz_min, mz_max = np.min(mz_lb), np.max(mz_ub)
    mob_min, mob_max = np.min(mob_lb), np.max(mob_ub)

    for i, file_dir in enumerate(line_list):
        with pymzml.run.Reader(file_dir, obo_version = '4.1.9') as reader:
            
            # Initialize data collector for the line
            num_spe = reader.get_spectrum_count()
            assert num_spe>0, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)

            line_pixels = np.zeros((num_spe, len(MS1_list)+1))
            line_acq_times = np.zeros((num_spe))

            for j, spec in tqdm(enumerate(reader), desc = "progress thru line {linenum} of {total_lines}".format(linenum = i+1, total_lines = len(line_list)), total = num_spe, delay = 0.05, disable = (testing or gui)):
                # Update gui variables            
                if gui:
                    tkinter_widgets[0]['value']=(100*i/len(line_list))+((100/len(line_list))*(j/num_spe))
                    tkinter_widgets[0].update()
                    tkinter_widgets[2]['text']=f'line {i+1}/{len(line_list)}, spectrum {j+1}/{num_spe}'
                    tkinter_widgets[2].update()

                # Get TIC and retention time
                line_acq_times[j] = spec.scan_time[0]
                line_pixels[j,0] = spec.TIC

                # get spectrum
                intensity_points = spec.i
                mzs = spec.mz

                # Get mobility values
                all_present_arrays = spec.get_all_arrays_in_spec()
                if "mean inverse reduced ion mobility array" in all_present_arrays:
                    mobs = spec.get_array("mean inverse reduced ion mobility array")
                else:
                    try:
                        with HiddenPrints():
                            mobs = spec.get_array("raw ion mobility array")
                    except:
                        mobs = None
                if mobs is None:
                    mobs = np.zeros(mzs.shape)

                # remove peaks with 0 intensity for faster slicing
                zeros_mask = np.where(intensity_points!=0)[0]
                # print(zeros_mask,type(intensity_points), type(mzs), type(mobs))
                intensity_points = intensity_points[zeros_mask]
                mzs = mzs[zeros_mask]
                mobs = mobs[zeros_mask]

                # remove peaks outside of potential mass and mobility ranges for faster slicing
                mz_mob_mask = np.where((mzs>mz_min)&(mzs<mz_max)&(mobs>mob_min)&(mobs<mob_max))
                intensity_points = intensity_points[mz_mob_mask]
                mzs = mzs[mz_mob_mask]
                mobs = mobs[mz_mob_mask]
                
                # Select each peak based on corresponding mass/mobility window
                for k in range(len(mz_lb)):
                    selected_idxs_mask = np.where((mzs>mz_lb[k])&(mzs<mz_ub[k])&(mobs>mob_lb[k])&(mobs<mob_ub[k]))
                    line_pixels[j,k+1] = np.sum(intensity_points[selected_idxs_mask])

        pixels_meta.append(line_pixels)
        acq_times.append(line_acq_times)


    metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
    metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

    pixels_aligned = msigen.ms1_interp(pixels_meta, acq_times, MS1_list, line_list)
    
    return metadata, pixels_aligned


# ======================================================================
# MS2 
# ======================================================================
def check_dim_mzml(line_list, experiment_type, ShowNumLineSpe=False):
    # determine the filetype given    
    acq_times = []
    filter_list = []

    for file_dir in line_list:
        with pymzml.run.Reader(file_dir, obo_version = '4.1.9') as reader:
            num_spe = reader.get_spectrum_count()
            assert num_spe>0, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)

            # Get Start times, end times, number of spectra per line, and list of unique filters.
            line_acq_times = []
            line_filter_list = []

            for spectrum in reader:
                # get mass fragmentation level as an integer
                level_int = spectrum.ms_level
                # Match formatting of other workflows
                if level_int == 1: level = 'MS1'
                else: level = 'MS2'

                # only check for precursors and collision energy if MS2 scan
                if level == 'MS2':
                    mz = spectrum.selected_precursors[0]['mz']
                    energy = spectrum['collision energy']
                else: 
                    mz = 0.0
                    energy = 0.0
                if energy == None: energy = 0.0

                # Check polarity
                if spectrum['positive scan'] == True:
                    polarity = '+'
                elif spectrum['negative scan'] == True:
                    polarity = '-'
                else:
                    polarity = ''
                
                # get retention time information
                line_acq_times.append(spectrum.scan_time[0])

                # Get mass range
                mass_range_start, mass_range_end = spectrum['scan window lower limit'], spectrum['scan window upper limit']
                if not (mass_range_start and mass_range_end):
                    mass_range_start, mass_range_end = spectrum.extremeValues('mz')

                # if no mobility present
                if experiment_type in [0,1]:
                    # save filter information
                    line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end])

                # if mobility present
                else:
                    mob_range_start, mob_range_end = get_mobility_range_from_mzml_spectrum(spectrum)
                    line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end, mob_range_start, mob_range_end])

        # warnings for using the incorrect is_ms2 and is_mobility values
        if any([i in ["mean inverse reduced ion mobility array", "raw ion mobility array"] for i in spectrum.get_all_arrays_in_spec()]):
            if experiment_type in [0,1]:
                raise Warning("The data file used contains mobility data but is_mobility is set to False")
        if 'MS2' in np.array(line_filter_list):
            if experiment_type in [0,2]:
                raise Warning("The data file used contains MS2 data but is_MS2 is set to False")
            
        filter_list.append(line_filter_list)
        acq_times.append(line_acq_times)
    
    num_spe_per_line = [len(i) for i in acq_times]
    # show results
    if ShowNumLineSpe:
        print('\nline scan spectra summary\n# of lines is: {}\nmean # of spectra is: {}\nmin # of spectra is: {}\nmean start time is {}\nmean end time is: {}'.format(
            len(num_spe_per_line), int(np.mean(num_spe_per_line)), int(np.min(num_spe_per_line)),np.mean([i[0] for i in acq_times]),np.mean([i[-1] for i in acq_times])))

    return acq_times, filter_list    

# TODO: merge with as get_ScansPerFilter_d

def get_ScansPerFilter_mzml(line_list, filters_info, all_filters_list, filter_inverse, display_tqdm = False):
    '''
    Works for multi Filters.
    '''
    # unpack filters_info
    filter_list = filters_info[0]

    # accumulator
    scans_per_filter = np.zeros((len(all_filters_list), len(filter_list)), dtype = int)
    
    # used to separate the filter_list into each line
    counter = 0
    for i in tqdm(range(len(all_filters_list)), disable = not display_tqdm):
        # Get each filter
        for j in filter_inverse[counter: counter + len(all_filters_list[i])]:            
            # count on
            scans_per_filter[i,j]+=1

    return scans_per_filter

# TODO: merge with as reorder_pixels_d

def reorder_pixels_mzml(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs):
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


# ======================================================================
# MS2 without mobility
# ======================================================================

def mzml_ms2_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes=True, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    # get mass windows

    MS1_list, _, MS1_polarity_list, prec_list, frag_list, _, MS2_polarity_list, mass_list_idxs = mass_lists

    if not gui:
        print("preprocessing data...")
        t_i = time()
    acq_times, all_filters_list = check_dim_mzml(line_list, experiment_type, ShowNumLineSpe=in_jupyter)

    metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
    metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

    filters_info, filter_inverse = get_filters_info_mzml(all_filters_list)
    PeakCountsPerFilter, mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter \
        = msigen.get_PeakCountsPerFilter(filters_info, mass_lists, lower_lims, upper_lims)
    # finds the number of scans that use a specific filter
    scans_per_filter = get_ScansPerFilter_mzml(line_list, filters_info, all_filters_list, filter_inverse)
    consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
        = msigen.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)
    num_filter_groups = len(consolidated_filter_list)

    # get an array that gives the scan group number from the index of any scan (1d index)
    grp_from_scan_idx = np.empty((len(filters_info[0])), dtype = int)
    for idx, i in enumerate(consolidated_idx_list):
        for j in i:
            grp_from_scan_idx[j]=idx
    grp_from_scan_idx = grp_from_scan_idx[filter_inverse]


    # There was an issue with the scans_per_filter_group defined above. This overwrites it because I couldnt figure out what the issue was above.
    scans_per_filter_grp = np.zeros((len(line_list), num_filter_groups), dtype = int)
    j=0
    for i in range(len(acq_times)):
        scans_per_filter_grp[i,:] = np.unique(grp_from_scan_idx[np.arange(j,j+len(acq_times[i]))], return_counts=True, axis = 0)[1]
        j+=len(acq_times[i])

    if not gui:        
        print("finished data preprocessing after {tot_time:.2f} s".format(tot_time = time()-t_i))

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    all_TimeStamps = []
    pixels_metas = []

    # holds index of current scan
    scan_idx = 0

    for i, Name in tqdm(enumerate(line_list), desc = 'Progress through lines', total = len(line_list), disable = (testing or gui)):        
        # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
        TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
        # counts how many times numbers have been inputted each array
        counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

        with pymzml.run.Reader(Name, obo_version = '4.1.9') as reader:
            # collect metadata from raw file
            # if i == 0:
            #     metadata = get_basic_instrument_metadata_raw_no_mob(data, metadata)

            # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
            pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]

            for j, spectrum in tqdm(enumerate(reader), disable = True):
                # Update gui variables            
                if gui:
                    tkinter_widgets[0]['value']=(100*i/len(line_list))+((100/len(line_list))*(j/len(acq_times[i])))
                    tkinter_widgets[0].update()
                    tkinter_widgets[2]['text']=f'line {i+1}/{len(line_list)}, spectrum {j+1}/{len(acq_times[i])}'
                    tkinter_widgets[2].update()

                # determine which group is going to be used
                grp = grp_from_scan_idx[scan_idx]
                counter[grp]+=1

                # handle info
                TimeStamps[grp][counter[grp]] = acq_times[i][j] 

                pixels_meta[grp][counter[grp], 0] = spectrum.TIC

                # skip filters with no masses in the mass list
                if peak_counts_per_filter_grp[grp]:

                    # get mz and intensity values
                    mz = spectrum.mz
                    intensity_points = np.append(spectrum.i,0)
                    
                    lbs,ubs = np.array(mzs_per_filter_grp_lb[grp]), np.array(mzs_per_filter_grp_ub[grp])
                
                    if msigen.numba_present:
                        idxs_to_sum = msigen.vectorized_sorted_slice_njit(mz, lbs, ubs)
                        pixel = msigen.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                        pixels_meta[grp][counter[grp],1:] = pixel
                    else:
                        idxs_to_sum = msigen.vectorized_sorted_slice(mz, lbs, ubs) # Slower
                        pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

                # keep count of the 1d scan index
                scan_idx += 1

        all_TimeStamps.append(TimeStamps)
        pixels_metas.append(pixels_meta)

    pixels, all_TimeStamps_aligned = msigen.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, normalize_img_sizes, mzs_per_filter_grp, line_list)

    # # Normalize timestamps to align each line in case one line took longer or started later.
    # all_TimeStamps_normed  = msigen.normalize_ms2_timestamps(all_TimeStamps, acq_times)

    # # Deterime how many pixels to use for each group of transitions and get evenly spaced times to sample at
    # num_spe_per_group_aligned = msigen.get_num_spe_per_group_aligned(scans_per_filter_grp, normalize_img_sizes)
    # all_TimeStamps_aligned = [np.linspace(0,1,i) for i in num_spe_per_group_aligned]

    # # make the final output of shape (lines, pixels_per_line, num_transitions+1)
    # pixels = [np.zeros((len(line_list), num_spe_per_group_aligned[i], len(mzs)+1)) for (i, mzs) in enumerate(mzs_per_filter_grp)]

    # # go through the extracted data and place them into pixels_final. list by group idx with shapes (# of lines, # of Pixels per line, m/z)
    # for i, pixels_meta in enumerate(pixels_metas):
    #     for j, pixels_meta_grp in enumerate(pixels_meta):
    #         points = (all_TimeStamps_normed[i][j], np.arange(pixels_meta_grp.shape[1]))
    #         sampling_points = np.array(np.meshgrid(*(all_TimeStamps_aligned[j], np.arange(pixels_meta_grp.shape[1])), indexing = 'ij')).transpose(1,2,0)
    #         pixels[j][i] = interpn(points, pixels_meta_grp, sampling_points, method = 'nearest', bounds_error = False, fill_value=None)

    # Order the pixels in the way the mass list csv/excel file was ordered
    pixels = reorder_pixels_mzml(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)  
    if normalize_img_sizes:
        pixels = msigen.pixels_list_to_array(pixels, line_list, all_TimeStamps_aligned)

    return metadata, pixels

# TODO: merge with as get_filters_info_d
def get_filters_info_mzml(all_filters_list):
    filter_list = []
    acq_polars = []
    acq_types = []
    precursors = []
    mz_ranges = []
    for i in all_filters_list:
        filter_list.extend(i)
    filter_list, filter_inverse = np.unique(filter_list, return_inverse=True, axis = 0)

    for (mz, energy, level, polarity, mass_range_start, mass_range_end) in filter_list:
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

    return [filter_list, acq_polars, acq_types, precursors, mz_ranges], filter_inverse


# TODO: merge with as get_filter_idx_d
def get_filter_idx_mzml(Filter,acq_types,acq_polars,mz_ranges,precursors):

    precursor, energy, acq_type, acq_polar, mass_range_start, mass_range_end = Filter

    if acq_polar == '+':
        polarity_numeric = 1.0
    elif acq_polar == '-':
        polarity_numeric = -1.0

    if acq_type == 'MS1':   # since filter name varies for ms, we just hard code this situation. 
        precursor = 0.0
        mz_range = [100.0, 950.0]
    elif acq_type == 'MS2':
        mz_range = [float(mass_range_start),float(mass_range_end)]
    
    mz_range_judge = np.array(mz_range).reshape(1, 2) == np.array(mz_ranges).astype(float)

    # to match look-up table: acq_types, acq_polars, precursors
    if acq_type == 'MS1':
        idx = (polarity_numeric == acq_polars)&(acq_type == acq_types)&(mz_range_judge[:,0])&(mz_range_judge[:,1])
    if acq_type == 'MS2': 
        idx = (polarity_numeric == acq_polars)&(acq_type == acq_types)&(mz_range_judge[:,0])&(mz_range_judge[:,1])&(precursor == precursors)
    idx = np.where(idx)[0]
    return idx


# ======================================================================
# MS2 with mobility
# ======================================================================

def mzml_ms2_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes=True, in_jupyter = True, testing = False, gui = False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = mass_lists
    if not gui:
        print("preprocessing data...")
        t_i = time()
    acq_times, all_filters_list = check_dim_mzml(line_list, experiment_type, ShowNumLineSpe=in_jupyter)

    metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
    metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

    filters_info, filter_inverse = get_filters_info_mob(all_filters_list)

    PeakCountsPerFilter, mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mobsPerFilter_lb, mobsPerFilter_ub, mzIndicesPerFilter \
        = msigen.get_PeakCountsPerFilter(filters_info, mass_lists, lower_lims, upper_lims)

    scans_per_filter = get_ScansPerFilter_mzml(line_list, filters_info, all_filters_list, filter_inverse)

    consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
        = msigen.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)

    #get ms level of each filter group
    ms_lvl_per_filter_grp = []
    for grp in consolidated_filter_list:
        ms_lvl_per_filter_grp.append(grp[0][2])

    num_filter_groups = len(consolidated_filter_list)

    # get an array that gives the scan group number from the index of any scan (1d index)
    grp_from_scan_idx = np.empty((len(filters_info[0])), dtype = int)
    for idx, i in enumerate(consolidated_idx_list):
        for j in i:
            grp_from_scan_idx[j]=idx
    grp_from_scan_idx = grp_from_scan_idx[filter_inverse]
    
    if not gui:
        print("finished data preprocessing after {tot_time:.2f} s".format(tot_time = time()-t_i))

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    all_TimeStamps = []
    pixels_metas = []

    # holds index of current scan/spectrum
    scan_idx = 0

    for i, file_dir in tqdm(enumerate(line_list), desc = 'Progress through lines', total = len(line_list), disable = (testing or gui)):        
        # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
        TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
        # counts how many times numbers have been inputted each array
        counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

        with pymzml.run.Reader(file_dir, obo_version = '4.1.9') as reader:
            # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
            pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
                        
            for j, spectrum in enumerate(reader):
                # Update gui variables            
                if gui:
                    tkinter_widgets[0]['value']=(100*i/len(line_list))+((100/len(line_list))*(j/len(acq_times[i])))
                    tkinter_widgets[0].update()
                    tkinter_widgets[2]['text']=f'line {i+1}/{len(line_list)}, spectrum {j+1}/{len(acq_times[i])}'
                    tkinter_widgets[2].update()

                # collect metadata from raw file
                # if i == 0:
                #     metadata = get_basic_instrument_metadata_raw_no_mob(data, metadata)

                # determine which group is going to be used
                grp = grp_from_scan_idx[scan_idx]
                counter[grp]+=1

                # handle info
                TimeStamps[grp][counter[grp]] = acq_times[i][j] 
                pixels_meta[grp][counter[grp], 0] = spectrum.TIC

                # skip filters with no masses in the mass list
                if peak_counts_per_filter_grp[grp]:

                    mz = spectrum.mz
                    intensity_points = np.append(spectrum.i,0)

                    # Get mobility values
                    all_present_arrays = spectrum.get_all_arrays_in_spec()
                    if "mean inverse reduced ion mobility array" in all_present_arrays:
                        mob = spectrum.get_array("mean inverse reduced ion mobility array")
                    else:
                        try:
                            with HiddenPrints():
                                mob = spectrum.get_array("raw ion mobility array")
                        except:
                            mob = None
                    if mob is None:
                        mob =[0.0]

                    # get all m/z and mobility values that bound the selection windows as their tof or scan index
                    lbs = np.array(mzs_per_filter_grp_lb[grp])
                    ubs = np.array(mzs_per_filter_grp_ub[grp])
                    mob_lbs = np.array(mobsPerFilter_lb[consolidated_idx_list[grp][0]])
                    mob_ubs = np.array(mobsPerFilter_ub[consolidated_idx_list[grp][0]])

                    # simultaneously slice by mz and mobility
                    idxs_to_sum = msigen.vectorized_unsorted_slice_mob(mz,mob,lbs,ubs,mob_lbs,mob_ubs)
                    pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, np.array(idxs_to_sum)), axis = 1)

                # keep count of the 1d scan index
                scan_idx += 1

        all_TimeStamps.append(TimeStamps)
        pixels_metas.append(pixels_meta)

    pixels, all_TimeStamps_aligned = msigen.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, normalize_img_sizes, mzs_per_filter_grp, line_list)

    # # Normalize timestamps to align each line in case one line took longer or started later.
    # all_TimeStamps_normed  = msigen.normalize_ms2_timestamps(all_TimeStamps, acq_times)

    # # Deterime how many pixels to use for each group of transitions and get evenly spaced times to sample at
    # num_spe_per_group_aligned = msigen.get_num_spe_per_group_aligned(scans_per_filter_grp, normalize_img_sizes)
    # all_TimeStamps_aligned = [np.linspace(0,1,i) for i in num_spe_per_group_aligned]

    # # make the final output of shape (lines, pixels_per_line, num_transitions+1)
    # pixels = [np.zeros((len(line_list), num_spe_per_group_aligned[i], len(mzs)+1)) for (i, mzs) in enumerate(mzs_per_filter_grp)]

    # # go through the extracted data and place them into pixels_final. list by group idx with shapes (# of lines, # of Pixels per line, m/z)
    # for i, pixels_meta in enumerate(pixels_metas):
    #     for j, pixels_meta_grp in enumerate(pixels_meta):
    #         points = (all_TimeStamps_normed[i][j], np.arange(pixels_meta_grp.shape[1]))
    #         sampling_points = np.array(np.meshgrid(*(all_TimeStamps_aligned[j], np.arange(pixels_meta_grp.shape[1])), indexing = 'ij')).transpose(1,2,0)
    #         pixels[j][i] = interpn(points, pixels_meta_grp, sampling_points, method = 'nearest', bounds_error = False, fill_value=None)

    # Order the pixels in the way the mass list csv/excel file was ordered
    pixels = reorder_pixels_mzml(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)
    if normalize_img_sizes:
        pixels = msigen.pixels_list_to_array(pixels, line_list, all_TimeStamps_aligned)

    return metadata, pixels    

def get_filters_info_mob(all_filters_list):
    filter_list = []
    acq_polars = []
    acq_types = []
    precursors = []
    mz_ranges = []
    mob_ranges = []
    for i in all_filters_list:
        filter_list.extend(i)
    filter_list, filter_inverse = np.unique(filter_list, return_inverse=True, axis = 0)

    for (mz, energy, level, polarity, mass_range_start, mass_range_end, mob_range_start, mob_range_end) in filter_list:
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
        mob_ranges.append([mob_range_start, mob_range_end])

    return [filter_list, acq_polars, acq_types, precursors, mz_ranges, mob_ranges], filter_inverse

def get_mobility_range_from_mzml_spectrum(spectrum):
    # get mobility range from keyword parameters if possible
    mob_range_start, mob_range_end = getUserParam(spectrum, 'ion mobility lower limit'), getUserParam(spectrum, 'ion mobility upper limit') 
    # otherwise extract the mobility array and get the max & min values.
    if (mob_range_start is None) or (mob_range_end is None):
        all_present_arrays = spectrum.get_all_arrays_in_spec()
        if "mean inverse reduced ion mobility array" in all_present_arrays:
            mob = spectrum.get_array("mean inverse reduced ion mobility array")
        else:
            try:
                with HiddenPrints():
                    mob = spectrum.get_array("raw ion mobility array")
            except:
                mob = None
        if mob is None:
            mob =[0.0]
        mob_range_start, mob_range_end = np.min(mob), np.max(mob)
    return [mob_range_start, mob_range_end]

def getUserParam(spectrum, accession):
    search_string = './/*[@name="{0}"]'.format(accession)
    elements = []
    for x in spectrum.element.iterfind(search_string):
        val = x.attrib.get("value", "")
        try:
            val = float(val)
        except:
            pass
        elements.append(val)

    if len(elements) == 0:
        return_val = None
    elif len(elements) == 1:
        return_val = elements[0]
    else:
        return_val = elements
    if return_val == "":
        return_val = True
    return return_val






    