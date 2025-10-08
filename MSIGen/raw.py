"""
This module contains a subclass of the base MSIGen class for handling files Thermo files with the .raw file extension.
This works for MS1 or MS2 data files without ion mobility.
"""

from MSIGen.base_class import MSIGen_base

# Agilent, Thermo data access
from multiplierz.mzAPI import mzFile

from tqdm import tqdm
import numpy as np
# from scipy.interpolate import interpn#, NearestNDInterpolator

class MSIGen_raw(MSIGen_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_files(self, *args, **kwargs):
        """Processes the data files based on the MS level and whether ion mobility data are present."""
        
        if not self.is_MS2 and not self.is_mobility:
            return self.ms1_no_mob(*args, **kwargs)
        elif self.is_MS2 and not self.is_mobility:
            return self.ms2_no_mob(*args, **kwargs)
        else:
            raise NotImplementedError('Processing .raw files with mobility data is not yet supported.')
        
    # ==================================
    # General functions
    # ==================================
    def get_basic_instrument_metadata(self, data, metadata = {}):
        """Gets some of the instrument metadata from the data file."""
        
        metadata_vars = ['filter_list']
        self.metadata = self.get_attr_values(self.metadata, data, metadata_vars)

        metadata_vars = ['CreationDate']
        source = data.source
        self.metadata = self.get_attr_values(self.metadata, source, metadata_vars)

        metadata_vars = ['Name','Model','HardwareVersion','SoftwareVersion','SerialNumber','IsTsqQuantumFile']
        inst_data = source.GetInstrumentData()
        self.metadata = self.get_attr_values(self.metadata, inst_data, metadata_vars)

        # Other parameters
        instrumental_values = []
        for i in data.scan_range():
            instrumental_values.append(i)
        #input into dict
        self.metadata['instrumental_values']=instrumental_values

        return self.metadata

    # ==================================
    # MS1 - No Mobility
    # ==================================
    def get_scan_without_zeros(self, data, scannum, centroid = False):
        """
        A faster implentation of multiplierz scan method for .raw files.
        
        Args:
            data: The mzFile object containing the raw data.
            scannum: The scan number to retrieve.
            centroid: Boolean indicating whether to use centroid data (True) or profile data (False). Default is False.

        Returns:
            mz (np.ndarray): The m/z values of the scan.
            intensity_points (np.ndarray): The intensity values of the scan.
        """

        scan_stats = data.source.GetScanStatsForScanNumber(scannum)
        # Does IsCentroidScan indicate that profile data is not available?
        if centroid or scan_stats.IsCentroidScan:
            
            stream = data.source.GetCentroidStream(scannum, False)
            if stream.Masses is not None and stream.Intensities is not None:
                return np.array(stream.Masses), np.array(stream.Intensities)
            else:
                # Fall back on "profile" mode, which seems to usually turn
                # out centroid data for some reason.  The format is confused.
                scan = data.source.GetSegmentedScanFromScanNumber(scannum, scan_stats) 
                return np.array(scan.Positions), np.array(scan.Intensities)
        
        else: # Profile-only scan.
            scan = data.source.GetSegmentedScanFromScanNumber(scannum, scan_stats)
            return np.array(scan.Positions), np.array(scan.Intensities)


    def ms1_no_mob(self, metadata={}, in_jupyter = None, testing = None, gui=None, pixels_per_line = None, tkinter_widgets = None, **kwargs):
        """
        Data processing for Thermo .raw files with only MS1 data.
        
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
        # unpack variables
        for i in [("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])

        # monitor progress on gui
        self.progressbar_start_preprocessing()

        # unpack variables
        MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = self.mass_list
        lb, _, _, _, _, _, _ = self.lower_lims
        ub, _, _, _, _, _, _ = self.upper_lims

        self.progressbar_start_extraction()

        pixels = []
        rts = []

        for i, file_dir in tqdm(enumerate(self.line_list), total = len(self.line_list), desc='Progress through lines', disable = self.testing or self.gui):
            
            # Get relevant data if .raw data.      
            data = mzFile(file_dir)

            headers = np.array(data.xic()) # Default parameters for xic

            # some files may use "SIM ms" instead of "Full ms". If this is the case, use "SIM ms" filters instead.
            # There will still be a failure or strange outputs if multiple SIM windows are used.
            if len(headers) == 0:
                headers = np.array(data.xic(filter = 'SIM ms'))

            assert len(headers)>0, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)
            assert headers.shape[1] == 2, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)

            Acq_times = np.round(headers[:,0], 4)
            num_spe = len(Acq_times)

            line_pixels = np.zeros((num_spe, MS1_list.shape[0]+1))

            # Get masses for each scan
            for j in tqdm(range(num_spe), desc = 'Progress through line {}'.format(i+1), disable = True):
                # Update line dependent gui variables            
                self.progressbar_update_progress(num_spe, i, j)

                # remove zeros from the arrays for faster slicing
                mz, intensity_points = self.get_scan_without_zeros(data, j+1, False)
                intensity_points_mask = np.where(intensity_points)
                intensity_points = np.append(intensity_points[intensity_points_mask[0]],0)
                mz = mz[intensity_points_mask[0]]            

                # get TIC
                line_pixels[j-1,0] = np.sum(intensity_points)

                pixel = self.extract_masses_no_mob(mz, lb, ub, intensity_points)
                line_pixels[j-1,1:] = pixel
                    
            data.close()
        
            pixels.append(line_pixels)
            rts.append(Acq_times)
        
        # Save average start and end retention times
        self.metadata['average_start_time'] = np.mean([i[0] for i in rts])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in rts])

        self.rts = rts
        # align number and time of pixels
        pixels_aligned = self.ms1_interp(pixels, mass_list = MS1_list)

        return self.metadata, pixels_aligned


    # ==================================
    # MS2 - No Mobility
    # ==================================
    def parse_filter_string(self, string):
        """Parses a Thermo filter string into a dictionary of its components.
        Args:
            string (str): The filter string to parse.
        Returns:
            dict: A dictionary containing the parsed components of the filter string.
        """
        match = self.thermo_scan_filter_string_patterns.match(string)
        # only keep values that were actually extracted
        outdict = {}
        for key in match.groupdict():
            if match.group(key) is not None:
                outdict[key] = match.group(key)
        return outdict

    def make_filter_string_from_filter_dict(self, filter_dict):
        """ Constructs a Thermo filter string from a dictionary of filter components.
        Args:
            filter_dict (dict): Dictionary containing filter components such as:
                analyzer, polarity, dataType, source, scanType, msMode, precursorMz, activationType, activationEnergy, scanRangeStart, scanRangeEnd.
        Returns:
            str: A formatted filter string.
        """
        parts = []
        if filter_dict.get("analyzer", None):
            parts.append(filter_dict["analyzer"])
        if filter_dict.get("segment", None) and filter_dict.get("event", None):
            parts.append("{{"+filter_dict["segment"]+","+filter_dict["event"]+"}}")
        if filter_dict.get("polarity", None):
            parts.append(filter_dict["polarity"])
        if filter_dict.get("dataType", None):
            parts.append(filter_dict["dataType"])
        if filter_dict.get("source", None):
            parts.append(filter_dict["source"])
        if filter_dict.get("corona", None):
            parts.append(filter_dict["corona"])
        if filter_dict.get("photoIonization", None):
            parts.append(filter_dict["photoIonization"])
        if filter_dict.get("sourceCID", None):
            parts.append(filter_dict["sourceCID"])
        if filter_dict.get("detectorSet", None):
            parts.append(filter_dict["detectorSet"])
        if filter_dict.get("compensationVoltage", None):
            parts.append(f'cv={filter_dict["compensationVoltage"]}')
        if filter_dict.get("rapid", None):
            parts.append(filter_dict["rapid"])
        if filter_dict.get("turbo", None):
            parts.append(filter_dict["turbo"])
        if filter_dict.get("enhanced", None):
            parts.append(filter_dict["enhanced"])
        if filter_dict.get("sps", None):
            parts.append(filter_dict["sps"])
        if filter_dict.get("dependent", None):
            parts.append(filter_dict["dependent"])
        if filter_dict.get("wideband", None):
            parts.append(filter_dict["wideband"])
        if filter_dict.get("ultra", None):
            parts.append(filter_dict["ultra"])
        if filter_dict.get("supplementalActivation", None):
            parts.append(filter_dict["supplementalActivation"])
        if filter_dict.get("accurateMass", None):
            parts.append(filter_dict["accurateMass"])
        if filter_dict.get("scanType", None):
            parts.append(filter_dict["scanType"])
        if filter_dict.get("lockmass", None):
            parts.append(filter_dict["lockmass"])
        if filter_dict.get("multiplex", None):
            parts.append(filter_dict["multiplex"])
        if filter_dict.get("msMode", None):
            parts.append(filter_dict["msMode"])
        if filter_dict.get("precursorMz", None):
            precursor = filter_dict["precursorMz"]
            if filter_dict.get("activationType", None) or filter_dict.get("activationEnergy", None):
                activation = "@"
                if filter_dict.get("activationType"):
                    activation += filter_dict["activationType"]
                if filter_dict.get("activationEnergy"):
                    activation += str(filter_dict["activationEnergy"])
                precursor += activation
            parts.append(precursor)
        # scan range
        if filter_dict.get("scanRangeStart", None) and filter_dict.get("scanRangeEnd", None):
            parts.append(f'[{filter_dict["scanRangeStart"]}-{filter_dict["scanRangeEnd"]}]')
        return " ".join(parts)

    def reorder_pixels(self, pixels, filters_grp_info, mz_idxs_per_filter, mass_list_idxs, filters_info = None):
        """Reorders the pixels to match the order of the mass list."""
        # get the scan type/level
        iterator = [] 
        for filter_grp in filters_grp_info:
            iterator.append(filters_info[2][np.where(filter_grp[0]==filters_info[0])])

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

    def check_dim(self, ShowNumLineSpe=False):
        """
        Gets the acquisition times and other information about each scan to 
        decide what mass list entries can be obtained from each scan.
        
        Returns:
            acq_times (list): A list of acquisition times for each line.
            filter_list (list): A list of information from the filter strings for each spectrum in each line.
        """

        acq_times = []
        filter_list = []

        # Get Start times, end times, number of spectra per line, and list of unique filters.
        for file_dir in self.line_list:
            # Get relevant data if .raw data.
            data = mzFile(file_dir)

            if self.is_MS2:
                rts_and_filters = np.array(data.filters())
                # Check if there is a rt and a filter in the filters data
                assert len(rts_and_filters)>0, 'No retention time or scan filter data from file {} was obtained'.format(file_dir)
                assert rts_and_filters.shape[1] == 2, 'Data from file {} is in an unexpected format. Should be shape (2, num_scans) and got shape {}'.format(file_dir, rts_and_filters.shape)
                acq_times.append(rts_and_filters[:,0].astype(float))
                filter_list.append(rts_and_filters[:,1])
            data.close()
    
        num_spe_per_line = [len(i) for i in acq_times]
        # show results
        if ShowNumLineSpe:
            print('\nline scan spectra summary\n# of lines is: {}\nmean # of spectra is: {}\nmin # of spectra is: {}\nmean start time is {}\nmean end time is: {}'.format(
                len(num_spe_per_line), int(np.mean(num_spe_per_line)), int(np.min(num_spe_per_line)),np.mean([i[0] for i in acq_times]),np.mean([i[-1] for i in acq_times])))

        return acq_times, filter_list
    
    def ms2_no_mob(self, metadata = {}, normalize_img_sizes = None, in_jupyter = None, testing = None, gui=None, pixels_per_line = None, tkinter_widgets = None, **kwargs):
        """
        Data processing for Thermo .raw files that contain MS2 data.
        
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

        # unpack variables
        for i in [("normalize_img_sizes", normalize_img_sizes), ("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])
        
        # monitor progress on gui
        self.progressbar_start_preprocessing()
        if self.in_jupyter and not self.gui:
            print("Preprocessing data...")
        
        MS1_list, _, MS1_polarity_list, prec_list, frag_list, _, MS2_polarity_list, mass_list_idxs = self.mass_list

        acq_times, all_filters_list = self.check_dim(ShowNumLineSpe = self.in_jupyter)
        self.metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])
        
        # for MSMS, extracts info from filters
        filters_info, filter_inverse = self.get_filters_info(all_filters_list)
        # Determines correspondance of MZs to filters
        mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter = self.get_CountsPerFilter(filters_info)
        # finds the number of scans that use a specific filter
        scans_per_filter = self.get_ScansPerFilter(filters_info, all_filters_list = all_filters_list)
        # Groups filters into groups containing the same mzs/transitions
        consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, \
            mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
            = self.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)
        num_filter_groups = len(consolidated_filter_list)
        # for i in consolidated_filter_list:
        #     print(i)
        # get an array that gives the scan group number from the index of any scan (1d index)
        grp_from_scan_idx = np.empty((len(filters_info[0])), dtype = int)
        for idx, i in enumerate(consolidated_idx_list):
            for j in i:
                grp_from_scan_idx[j]=idx
        grp_from_scan_idx = grp_from_scan_idx[filter_inverse]

        # monitor progress on gui
        self.progressbar_start_extraction()

        all_TimeStamps = []
        pixels_metas = []
        
        # holds index of current scan
        scan_idx = 0

        for i, Name in tqdm(enumerate(self.line_list), desc = 'Progress through lines', total = len(self.line_list), disable = (self.testing or self.gui)):
            # accumulators for all filters,for line before interpolation, interpolation: intensity, scan/acq_time
            TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
            # counts how many times numbers have been inputted each array
            counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling
            
            if Name.lower().endswith('.raw'):
                data = mzFile(Name)
                
                # # collect metadata from raw file
                # if i == 0:
                #     metadata = get_basic_instrument_metadata_raw_no_mob(RawFile, metadata)

                # The intensity values for all masses/transitions in the mass list. 0 index in each group = TIC.
                pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]

                # counts how many times numbers have been inputted each array
                counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

                for j, TimeStamp in tqdm(enumerate(acq_times[i]), disable = True):
                    # Update gui variables
                    self.progressbar_update_progress(len(acq_times[i]), i, j)

                    # determine which group is going to be used
                    grp = grp_from_scan_idx[scan_idx]
                    counter[grp]+=1

                    # handle info
                    TimeStamps[grp][counter[grp]] = TimeStamp 

                    # get spectrum
                    mz, intensity_points = self.get_scan_without_zeros(data, j+1, False)

                    # get TIC
                    pixels_meta[grp][counter[grp], 0] = np.sum(intensity_points)

                    # skip filters with no masses in the mass list
                    if peak_counts_per_filter_grp[grp]:

                        # remove all values of zero to improve speed
                        intensity_points_mask = np.where(intensity_points)
                        mz = mz[intensity_points_mask[0]]
                        intensity_points = np.append(intensity_points[intensity_points_mask[0]],0)
                        
                        lbs,ubs = mzs_per_filter_grp_lb[grp], mzs_per_filter_grp_ub[grp] 

                        # TODO: Get this to work with the numba workflow
                        ### did not work properly with numba
                        # if self.numba_present:
                        #     idxs_to_sum = self.vectorized_sorted_slice_njit(mz, lbs, ubs)
                        #     pixel = self.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                        #     pixels_meta[grp][counter[grp],1:] = pixel
                        # else:
                        idxs_to_sum = self.vectorized_sorted_slice(mz, lbs, ubs) # Slower
                        pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

                    # keep count of the 1d scan index
                    scan_idx += 1

                data.close()

            all_TimeStamps.append(TimeStamps)
            pixels_metas.append(pixels_meta)

        self.rts = acq_times
        pixels, all_TimeStamps_aligned = self.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, mzs_per_filter_grp)

        # Order the pixels in the way the mass list csv/excel file was ordered
        pixels = self.reorder_pixels(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs, filters_info)    
        if normalize_img_sizes:
            pixels = self.pixels_list_to_array(pixels, all_TimeStamps_aligned)

        return self.metadata, pixels 

    def get_filters_info(self, filter_list):
        """
        Gets information about all filters present in the experiment.
        
        Returns:
            filters_info (list): A list of filter information, including filter names, polarities, MS levels, precursors, and mass ranges.
            polar_loc (int): The index of the polarity in the filter string.
            types_loc (list): A list of indices for the acquisition types in the filter string.
            filter_inverse (np.ndarray): An array of indices for the filters.
        """
        acq_polars = [] # + or -
        acq_types = [] # ms or ms2
        mz_ranges = [] # mass window
        precursors = [] # ms -> 0, ms2 -> precursor m/z

        filter_list, filter_inverse = np.unique([i for i in self.flatten_list(filter_list)], return_inverse=True)   # remove extra repeated ms1 filters
        
        for Filter in filter_list:
            # parse filter string
            filter_info = self.parse_filter_string(Filter)

            acq_polars.append(filter_info.get('polarity', None))
            try:
                # there must be a scanType and msMode in the filter string so it will throw an error if not present
                acq_types.append(filter_info.get('scanType', None) + " " + filter_info.get('msMode', None) + filter_info.get('msLevel', ''))
            except Exception as e:
                raise ValueError(f"Error parsing scan type and mode from filter: {Filter}. Error: {e}")
            mz_ranges.append([float(filter_info.get('scanRangeStart', 0.0)), float(filter_info.get('scanRangeEnd', 1000.0))])
            precursors.append(float(filter_info.get('precursorMz', 0.0)))
            
        acq_types, acq_polars, precursors, mz_ranges = np.array(acq_types), np.array(acq_polars), np.array(precursors), np.array(mz_ranges)

        return [filter_list, acq_polars, acq_types, precursors, mz_ranges], filter_inverse
    

    def get_ScansPerFilter(self, filters_info, all_filters_list, display_tqdm = False):
        """Determines the number of scans that use a specific filter group"""

        assert all_filters_list is not None, 'all_filters_list must be provided to get_ScansPerFilter'

        # unpack filters_info
        filter_list, acq_polars, acq_types, precursors, mz_ranges = filters_info

        # accumulator
        scans_per_filter = np.empty(([0, filter_list.shape[0]])).astype(int)

        for i, Name in tqdm(enumerate(self.line_list), disable = not display_tqdm):
            # counter for a line
            Dims = np.zeros((filter_list.shape[0])).astype(int)

            if Name.lower().endswith('.raw'):
                # Get each filter
                for j in range(len(all_filters_list[i])):
                    Filter = all_filters_list[i][j]

                    # Get the filter index of the scan
                    idx = np.where(Filter == filter_list)

                    # Add one to the count for this filter
                    Dims[idx] += 1

                # count on
                scans_per_filter = np.append(scans_per_filter, Dims.reshape((1, acq_polars.shape[0])), axis=0)

        return scans_per_filter