"""
This module contains a subclass of the base MSIGen class for handling files with the .mzml file extension.
This can handle files with or without ion mobility data, and with or without MS2 data.
This has been tested on the following file formats converted using MSConvert.
    Thermo .raw files that contain MS1 or MS2 data and do not contain ion mobility data.
    Agilent .d files containing MS1 or MS2 data with or without ion mobility data.
    Bruker .d files of .tsf format containing MS1 or MS2 data.
    Bruker .d files of .baf format containing MS1 data.
    Bruker .d files of .tdf format containing ion mobility data and MS1 or MS2 data.
.mzml files from other sources may or may not be processed as expected.
"""

from MSIGen.base_class import MSIGen_base

# mzML access
import pymzml

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interpn
from time import time

class MSIGen_mzml(MSIGen_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_files(self, *args, **kwargs):
        """Processes the data files based on the MS level and whether ion mobility data are present."""
        if (not self.is_MS2) and (not self.is_mobility):
            return self.ms1_no_mob(*args, **kwargs)
        elif (self.is_MS2) and (not self.is_mobility):
            return self.ms2_no_mob(*args, **kwargs)
        elif (not self.is_MS2) and (self.is_mobility):
            return self.ms1_mob(*args, **kwargs)
        else:
            return self.ms2_mob(*args, **kwargs)
            
    # ======================================================================
    # MS1 without mobility 
    # ======================================================================

    def ms1_no_mob(self, metadata=None, in_jupyter = None, testing = None, gui=None, pixels_per_line = None, tkinter_widgets = None, **kwargs):
        """
        Data processing for .mzml files with only MS1 data.
        
        Args:
            metadata (dict): Metadata dictionary to store instrument information. Overwrites self.metadata if provided.
            in_jupyter (bool): Flag indicating if the code is running in a Jupyter notebook. Overwrites self.in_jupyter if provided.
            testing (bool): Flag for testing mode. Overwrites self.testing if provided.
            gui (bool): Flag for GUI mode. Overwrites self.gui if provided.
            pixels_per_line (int): Number of pixels per line for the output image. Overwrites self.pixels_per_line if provided.
            tkinter_widgets: Tkinter widgets for GUI progress bar. Overwrites self.tkinter_widgets if provided.
        
        Returns:
            metadata (dict): Updated metadata dictionary with instrument information.
            pixels_aligned (np.ndarray): 3D array of intensity data of shape (m/z+1, lines, pixels_per_line).
        """

        # unpack variables. Any other kwargs are ignored.
        for i in [("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets), ("metadata", metadata)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])

        # monitor progress on gui
        self.progressbar_start_preprocessing()

        # get mass windows
        MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = self.mass_list
        lb, _, _, _, _, _, _ = self.lower_lims
        ub, _, _, _, _, _, _ = self.upper_lims

        # monitor progress on gui
        self.progressbar_start_extraction()
        
        # initiate accumulator
        pixels = []
        rts = []

        for i, file_dir in tqdm(enumerate(self.line_list), total = len(self.line_list), desc='Progress through lines', disable = (self.testing or self.gui)): 
            
            with pymzml.run.Reader(file_dir, obo_version = '4.1.9') as reader:

                # if i == 0:
                #     self.metadata = get_basic_instrument_metadata_mzml_no_mob(line_list, data, self.metadata)

                # grab headers for all scans
                num_spe = reader.get_spectrum_count()
                assert num_spe>0, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)

                line_rts = np.zeros(num_spe)
                line_pixels = np.zeros((num_spe, len(lb)+1))
                
                for j, spectrum in enumerate(reader):
                    # Update gui variables
                    self.progressbar_update_progress(num_spe, i, j)

                    # save scan time and TICs
                    line_rts[j] = spectrum.scan_time[0]
                    TIC = spectrum.TIC
                    line_pixels[j,0] = TIC

                    # get mz and intensity values
                    mz = spectrum.mz
                    intensity_points = np.append(spectrum.i,0)

                    pixel = self.extract_masses_no_mob(mz, lb, ub, intensity_points)
                    line_pixels[j,1:] = pixel

            pixels.append(line_pixels)
            rts.append(line_rts)

        self.metadata['average_start_time'] = np.mean([i[0] for i in rts])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in rts])

        self.rts = rts
        pixels_aligned = self.ms1_interp(pixels, mass_list = MS1_list)
        
        return self.metadata, pixels_aligned

    # ======================================================================
    # MS1 with mobility 
    # ======================================================================

    # MAKE SURE MOBILITY DATA ARE COMBINED WHEN USING MSCONVERT
    def mzml_ms1_mob(self, metadata=None, in_jupyter = None, testing = None, gui=None, pixels_per_line = None, tkinter_widgets = None, **kwargs):
        """
        Data processing from .mzml files with only MS1 data and ion mobility data.
        When using MSConvert to create this .mzml file, the option "combine ion mobility scans" must be checked for MSIGen to read the data properly.
        
        Args:
            metadata (dict): Metadata dictionary to store instrument information. Overwrites self.metadata if provided.
            in_jupyter (bool): Flag indicating if the code is running in a Jupyter notebook. Overwrites self.in_jupyter if provided.
            testing (bool): Flag for testing mode. Overwrites self.testing if provided.
            gui (bool): Flag for GUI mode. Overwrites self.gui if provided.
            pixels_per_line (int): Number of pixels per line for the output image. Overwrites self.pixels_per_line if provided.
            tkinter_widgets: Tkinter widgets for GUI progress bar. Overwrites self.tkinter_widgets if provided.
        
        Returns:
            metadata (dict): Updated metadata dictionary with instrument information.
            pixels_aligned (np.ndarray): 3D array of intensity data of shape (m/z+1, lines, pixels_per_line).
        """

        # unpack variables. Any other kwargs are ignored.
        for i in [("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets), ("metadata", metadata)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])

        # monitor progress on gui
        self.progressbar_start_preprocessing()

        MS1_list, MS1_mob_list, MS1_polarity_list, _, _, _, _, mass_list_idxs = self.mass_list

        # monitor progress on gui
        self.progressbar_start_extraction()

        pixels_meta = []
        rts = []

        # Get the mass ranges and determine the minimum and max mass and mobility values that may be used.
        mz_lb, mz_ub = self.lower_lims[0], self.upper_lims[0]
        mob_lb, mob_ub = self.lower_lims[1], self.upper_lims[1]
        mz_min, mz_max = np.min(mz_lb), np.max(mz_ub)
        mob_min, mob_max = np.min(mob_lb), np.max(mob_ub)

        for i, file_dir in enumerate(self.line_list):
            with pymzml.run.Reader(file_dir, obo_version = '4.1.9') as reader:
                
                # Initialize data collector for the line
                num_spe = reader.get_spectrum_count()
                assert num_spe>0, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)

                line_pixels = np.zeros((num_spe, len(MS1_list)+1))
                line_acq_times = np.zeros((num_spe))

                for j, spec in tqdm(enumerate(reader), desc = "progress thru line {linenum} of {total_lines}".format(linenum = i+1, total_lines = len(self.line_list)), total = num_spe, delay = 0.05, disable = (self.testing or self.gui)):
                    # Update gui variables
                    self.progressbar_update_progress(num_spe, i, j)

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
                            with self.HiddenPrints():
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
            rts.append(line_acq_times)


        self.metadata['average_start_time'] = np.mean([i[0] for i in rts])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in rts])

        self.rts = rts
        pixels_aligned = self.ms1_interp(pixels_meta, mass_list = MS1_list)
        
        return self.metadata, pixels_aligned


    # ======================================================================
    # MS2 
    # ======================================================================
    def check_dim(self, ShowNumLineSpe=False):
        """
        Gets the acquisition times and other information about each scan to 
        decide what mass list entries can be obtained from each scan.
        
        Returns:
            acq_times (list): A list of acquisition times for each line.
            filter_list (list): A list of information that would be included in Thermo-style filter strings for each line.
        """

        # determine the filetype given    
        acq_times = []
        filter_list = []

        for file_dir in self.line_list:
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
                    if not self.is_mobility:
                        # save filter information
                        line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end])

                    # if mobility present
                    else:
                        mob_range_start, mob_range_end = self.get_mobility_range_from_mzml_spectrum(spectrum)
                        line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end, mob_range_start, mob_range_end])

            # warnings for using the incorrect is_ms2 and is_mobility values
            if any([i in ["mean inverse reduced ion mobility array", "raw ion mobility array"] for i in spectrum.get_all_arrays_in_spec()]):
                if not self.is_mobility:
                    raise Warning("The data file used contains mobility data but is_mobility is set to False")
            if 'MS2' in np.array(line_filter_list):
                if not self.is_MS2:
                    raise Warning("The data file used contains MS2 data but is_MS2 is set to False")
                
            filter_list.append(line_filter_list)
            acq_times.append(line_acq_times)
        
        num_spe_per_line = [len(i) for i in acq_times]
        # show results
        if ShowNumLineSpe:
            print('\nline scan spectra summary\n# of lines is: {}\nmean # of spectra is: {}\nmin # of spectra is: {}\nmean start time is {}\nmean end time is: {}'.format(
                len(num_spe_per_line), int(np.mean(num_spe_per_line)), int(np.min(num_spe_per_line)),np.mean([i[0] for i in acq_times]),np.mean([i[-1] for i in acq_times])))

        return acq_times, filter_list

    def get_ScansPerFilter(self, filters_info, all_filters_list, filter_inverse, display_tqdm = False):
        """Determines the number of scans that use a specific filter group"""
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


    # ======================================================================
    # MS2 without mobility
    # ======================================================================

    def mzml_ms2_no_mob(self, metadata=None, normalize_img_sizes=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None, **kwargs):
        """
        Data processing for .mzml files that contain MS2 data.
        
        Args:
            metadata (dict): Metadata dictionary to store instrument information. Overwrites self.metadata if provided.
            normalize_img_sizes (bool): Flag indicating if image sizes should be normalized. Overwrites self.normalize_img_sizes if provided.
            in_jupyter (bool): Flag indicating if the code is running in a Jupyter notebook. Overwrites self.in_jupyter if provided.
            testing (bool): Flag for testing mode. Overwrites self.testing if provided.
            gui (bool): Flag for GUI mode. Overwrites self.gui if provided.
            pixels_per_line (int): Number of pixels per line for the output image. Overwrites self.pixels_per_line if provided.
            tkinter_widgets: Tkinter widgets for GUI progress bar. Overwrites self.tkinter_widgets if provided.
        
        Returns:
            metadata (dict): Updated metadata dictionary with instrument information.
            pixels_aligned (np.ndarray): 3D array of intensity data of shape (m/z+1, lines, pixels_per_line) or list of ion image arrays of shape (height, width).
        """

        # unpack variables. Any other kwargs are ignored.
        for i in [("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets), ("normalize_img_sizes", normalize_img_sizes), ("metadata", metadata)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])

        # monitor progress on gui
        self.progressbar_start_preprocessing()
        if not gui:
            print("preprocessing data...")
            t_i = time()

        # get mass windows
        MS1_list, _, MS1_polarity_list, prec_list, frag_list, _, MS2_polarity_list, mass_list_idxs = self.mass_list

        acq_times, all_filters_list = self.check_dim(ShowNumLineSpe=in_jupyter)

        self.metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

        filters_info, filter_inverse = self.get_filters_info(all_filters_list)
        mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter = self.get_CountsPerFilter(filters_info)
        # finds the number of scans that use a specific filter
        scans_per_filter = self.get_ScansPerFilter(filters_info, all_filters_list, filter_inverse)
        consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, \
            scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
            = self.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)
        num_filter_groups = len(consolidated_filter_list)

        # get an array that gives the scan group number from the index of any scan (1d index)
        grp_from_scan_idx = np.empty((len(filters_info[0])), dtype = int)
        for idx, i in enumerate(consolidated_idx_list):
            for j in i:
                grp_from_scan_idx[j]=idx
        grp_from_scan_idx = grp_from_scan_idx[filter_inverse]

        # There was an issue with the scans_per_filter_group defined above. This overwrites it because I couldnt figure out what the issue was above.
        scans_per_filter_grp = np.zeros((len(self.line_list), num_filter_groups), dtype = int)
        j=0
        for i in range(len(acq_times)):
            scans_per_filter_grp[i,:] = np.unique(grp_from_scan_idx[np.arange(j,j+len(acq_times[i]))], return_counts=True, axis = 0)[1]
            j+=len(acq_times[i])

        # monitor progress on gui
        self.progressbar_start_extraction()
        if not gui:
            print("finished data preprocessing after {tot_time:.2f} s".format(tot_time = time()-t_i))

        all_TimeStamps = []
        pixels_metas = []

        # holds index of current scan
        scan_idx = 0

        for i, Name in tqdm(enumerate(self.line_list), desc = 'Progress through lines', total = len(self.line_list), disable = (testing or gui)):
            # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
            TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
            # counts how many times numbers have been inputted each array
            counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

            with pymzml.run.Reader(Name, obo_version = '4.1.9') as reader:
                # collect metadata from raw file
                # if i == 0:
                #     self.metadata = get_basic_instrument_metadata_raw_no_mob(data, self.metadata)

                # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
                pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]

                for j, spectrum in tqdm(enumerate(reader), disable = True):
                    # Update gui variables
                    self.progressbar_update_progress(len(acq_times[i]), i, j)

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
                        
                        lb,ub = np.array(mzs_per_filter_grp_lb[grp]), np.array(mzs_per_filter_grp_ub[grp])
                    
                        pixel = self.extract_masses_no_mob(mz, lb, ub, intensity_points)
                        pixels_meta[grp][counter[grp],1:] = pixel

                        # if self.numba_present:
                        #     idxs_to_sum = self.vectorized_sorted_slice_njit(mz, lb, ub)
                        #     pixel = self.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                        #     pixels_meta[grp][counter[grp],1:] = pixel
                        # else:
                        #     idxs_to_sum = self.vectorized_sorted_slice(mz, lb, ub) # Slower
                        #     pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

                    # keep count of the 1d scan index
                    scan_idx += 1

            all_TimeStamps.append(TimeStamps)
            pixels_metas.append(pixels_meta)

        self.rts = acq_times
        pixels, all_TimeStamps_aligned = self.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, mzs_per_filter_grp)

        # Order the pixels in the way the mass list csv/excel file was ordered
        pixels = self.reorder_pixels(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)  
        if self.normalize_img_sizes:
            pixels = self.pixels_list_to_array(pixels, all_TimeStamps_aligned)

        return self.metadata, pixels

    ## Currently unused so commented out until sure it can be deleted
    # def get_filter_idx(self, Filter,acq_types,acq_polars,mz_ranges,precursors):

    #     precursor, energy, acq_type, acq_polar, mass_range_start, mass_range_end = Filter

    #     if acq_polar == '+':
    #         polarity_numeric = 1.0
    #     elif acq_polar == '-':
    #         polarity_numeric = -1.0

    #     if acq_type == 'MS1':   # since filter name varies for ms, we just hard code this situation. 
    #         precursor = 0.0
    #         mz_range = [100.0, 950.0]
    #     elif acq_type == 'MS2':
    #         mz_range = [float(mass_range_start),float(mass_range_end)]
        
    #     mz_range_judge = np.array(mz_range).reshape(1, 2) == np.array(mz_ranges).astype(float)

    #     # to match look-up table: acq_types, acq_polars, precursors
    #     if acq_type == 'MS1':
    #         idx = (polarity_numeric == acq_polars)&(acq_type == acq_types)&(mz_range_judge[:,0])&(mz_range_judge[:,1])
    #     if acq_type == 'MS2': 
    #         idx = (polarity_numeric == acq_polars)&(acq_type == acq_types)&(mz_range_judge[:,0])&(mz_range_judge[:,1])&(precursor == precursors)
    #     idx = np.where(idx)[0]
    #     return idx


    # ======================================================================
    # MS2 with mobility
    # ======================================================================

    def mzml_ms2_mob(self, metadata=None, normalize_img_sizes=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None, **kwargs):
        """
        Data processing from .mzml files that contain MS2 data and ion mobility data.
        
        Args:
            metadata (dict): Metadata dictionary to store instrument information. Overwrites self.metadata if provided.
            normalize_img_sizes (bool): Flag indicating if image sizes should be normalized. Overwrites self.normalize_img_sizes if provided.
            in_jupyter (bool): Flag indicating if the code is running in a Jupyter notebook. Overwrites self.in_jupyter if provided.
            testing (bool): Flag for testing mode. Overwrites self.testing if provided.
            gui (bool): Flag for GUI mode. Overwrites self.gui if provided.
            pixels_per_line (int): Number of pixels per line for the output image. Overwrites self.pixels_per_line if provided.
            tkinter_widgets: Tkinter widgets for GUI progress bar. Overwrites self.tkinter_widgets if provided.
        
        Returns:
            metadata (dict): Updated metadata dictionary with instrument information.
            pixels_aligned (np.ndarray): 3D array of intensity data of shape (m/z+1, lines, pixels_per_line) or list of ion image arrays of shape (height, width).
        """

        # unpack variables. Any other kwargs are ignored.
        for i in [("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets), ("normalize_img_sizes", normalize_img_sizes), ("metadata", metadata)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])

        # monitor progress on gui
        self.progressbar_start_preprocessing()
        if not gui:
            print("preprocessing data...")
            t_i = time()

        MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = self.mass_list
        acq_times, all_filters_list = self.check_dim(ShowNumLineSpe=in_jupyter)

        self.metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

        filters_info, filter_inverse = self.get_filters_info(all_filters_list)

        mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mobsPerFilter_lb, mobsPerFilter_ub, mzIndicesPerFilter \
            = self.get_CountsPerFilter(filters_info)

        scans_per_filter = self.get_ScansPerFilter(filters_info, all_filters_list, filter_inverse)

        consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, \
            scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
            = self.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)

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
        
        # monitor progress on gui
        self.progressbar_start_extraction()
        if not self.gui:
            print("finished data preprocessing after {tot_time:.2f} s".format(tot_time = time()-t_i))

        all_TimeStamps = []
        pixels_metas = []

        # holds index of current scan/spectrum
        scan_idx = 0

        for i, file_dir in tqdm(enumerate(self.line_list), desc = 'Progress through lines', total = len(self.line_list), disable = (testing or gui)):        
            # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
            TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
            # counts how many times numbers have been inputted each array
            counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

            with pymzml.run.Reader(file_dir, obo_version = '4.1.9') as reader:
                # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
                pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
                            
                for j, spectrum in enumerate(reader):
                    # Update gui variables
                    self.progressbar_update_progress(len(acq_times[i]), i, j)

                    # collect metadata from raw file
                    # if i == 0:
                    #     self.metadata = get_basic_instrument_metadata_raw_no_mob(data, self.metadata)

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
                                with self.HiddenPrints():
                                    mob = spectrum.get_array("raw ion mobility array")
                            except:
                                mob = None
                        if mob is None:
                            mob = [0.0]

                        # get all m/z and mobility values that bound the selection windows as their tof or scan index
                        lbs = np.array(mzs_per_filter_grp_lb[grp])
                        ubs = np.array(mzs_per_filter_grp_ub[grp])
                        mob_lbs = np.array(mobsPerFilter_lb[consolidated_idx_list[grp][0]])
                        mob_ubs = np.array(mobsPerFilter_ub[consolidated_idx_list[grp][0]])

                        # simultaneously slice by mz and mobility
                        idxs_to_sum = self.vectorized_unsorted_slice_mob(mz,mob,lbs,ubs,mob_lbs,mob_ubs)
                        pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, np.array(idxs_to_sum)), axis = 1)

                    # keep count of the 1d scan index
                    scan_idx += 1

            all_TimeStamps.append(TimeStamps)
            pixels_metas.append(pixels_meta)

        self.rts = acq_times
        pixels, all_TimeStamps_aligned = self.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, mzs_per_filter_grp)

        # Order the pixels in the way the mass list csv/excel file was ordered
        pixels = self.reorder_pixels(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)
        if normalize_img_sizes:
            pixels = self.pixels_list_to_array(pixels, all_TimeStamps_aligned)

        return self.metadata, pixels

    def get_mobility_range_from_mzml_spectrum(self, spectrum):
        """
        Determines the lower and upper bounds of the mobility range from the spectrum object.
        """
        # get mobility range from keyword parameters if possible
        mob_range_start, mob_range_end = self.getUserParam(spectrum, 'ion mobility lower limit'), self.getUserParam(spectrum, 'ion mobility upper limit') 
        # otherwise extract the mobility array and get the max & min values.
        if (mob_range_start is None) or (mob_range_end is None):
            all_present_arrays = spectrum.get_all_arrays_in_spec()
            if "mean inverse reduced ion mobility array" in all_present_arrays:
                mob = spectrum.get_array("mean inverse reduced ion mobility array")
            else:
                try:
                    with self.HiddenPrints():
                        mob = spectrum.get_array("raw ion mobility array")
                except:
                    mob = None
            if mob is None:
                mob =[0.0]
            mob_range_start, mob_range_end = np.min(mob), np.max(mob)
        return [mob_range_start, mob_range_end]

    def getUserParam(self, spectrum, param_name):
        """
        Obtains the value of a parameter based on its parameter name from the spectrum object.
        """
        search_string = './/*[@name="{0}"]'.format(param_name)
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

