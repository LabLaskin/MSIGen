from MSIGen import msigen

import os, sys
import numpy as np
from tqdm import tqdm

try:
    from MSIGen import tsf
    assert "dll" in dir(tsf)
except:
    print("Cannot extract Bruker .tsf data. Check that you input the timsdata.dll in the correct location")

# bruker tdf
try:
    from opentimspy.opentims import OpenTIMS
except:
    print('Could not import opentimspy. Cannot process .tdf format data from Bruker TIMS-TOF')

# Agilent
try:
    # import multiplierz and necessary dlls
    import multiplierz
    from multiplierz.mzAPI import mzFile
    import clr
    # gets dlls from within multiplierz package to access DesiredMSStorageType variable
    multiplierz_path = multiplierz.__file__
    dllpath = os.path.split(multiplierz_path)[0]+"\\mzAPI\\agilentdlls"
    sys.path += [dllpath]
    dlls = ['MassSpecDataReader', 'BaseCommon', 'BaseDataAccess']
    for dll in dlls: clr.AddReference(dll)
    import Agilent
    from Agilent.MassSpectrometry.DataAnalysis import DesiredMSStorageType

except: 
    print("Could not import mzFile. Cannot process Agilent's .d format data")


# =====================================================================================
# General functions
# =====================================================================================

def determine_file_format(line_list):
    MS_level = 'Not specified'
    if os.path.exists(os.path.join(line_list[0], 'analysis.tsf')):
        data_format = "bruker_tsf"
    elif os.path.exists(os.path.join(line_list[0], 'analysis.tdf')):
        data_format = "bruker_tdf"
    else:
        try:
            with HiddenPrints():
                data = mzFile(line_list[0])
            # vendor format
            data_format = data.format
            # MS1 or MS2
            MS_levels = np.unique(np.array(data.scan_info())[:,3])
            if 'MS2' in MS_levels:
                MS_level = 'MS2'
            else:
                MS_level = 'MS1'
            data.close()
        except:
            raise RuntimeError("Data file could not be read")

    return data_format, MS_level

def check_dim_d(line_list, experiment_type, ShowNumLineSpe=False):
    """Gets the times and other information about each scan to decide 
    what peaks can be obtained from each scan."""
    # determine the filetype given
    data_format, MS_level = determine_file_format(line_list)
    
    acq_times = []
    filter_list = []

    for file_dir in line_list:
        line_acq_times = []
        line_filter_list = []

        if data_format.lower() == 'agilent':
            # Get Start times, end times, number of spectra per line, and list of unique filters.
            with HiddenPrints():
                data = mzFile(file_dir)

            for rt, mz, index, level, polarity in data.scan_info():

                scanObj = data.source.GetSpectrum(data.source, index)
                rangeobj = scanObj.MeasuredMassRange
                mass_range_start = round(rangeobj.Start,1)
                mass_range_end = round(rangeobj.End,1)

                energy = float(scanObj.CollisionEnergy)

                line_acq_times.append(rt)
                line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end])
            data.close()
        
        elif data_format.lower() == 'bruker_tsf':
            data = tsf.tsf_data(file_dir, tsf.dll)
            
            msms_info = data.framemsmsinfo
            frames_info = data.frames.values
            for idx, rt, polarity, scanmode, msms_type, _, max_intensity, tic, num_peaks, MzCalibration, _,_,_ in frames_info:
                
                # get the values necessary for filters 
                [[mz, energy]] = msms_info[msms_info["Frame"]==1][['TriggerMass', 'CollisionEnergy']].values
                mass_range_start, mass_range_end = data.index_to_mz(idx, [0, len(data.read_profile_spectrum(idx))])
                level = parse_bruker_scan_level(scanmode)
                # save those values
                line_acq_times.append(rt)
                line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end])
            del data

        elif data_format.lower() == 'bruker_tdf':
            # column keys to be used in data.query
            columns_for_check_dim = ('mz','inv_ion_mobility')

            data = OpenTIMS(file_dir)

            # get relevant data for MSMS frames. 
            ## Can be saved potentially in 4 different tables with different keywords for the precursor mass.
            potential_tables_to_use = ['FrameMsMsInfo','PasefFrameMsMsInfo','DiaFrameMsMsWindows','PrmFrameMsMsInfo']
            potential_msms_precursor_keys = ["TriggerMass","IsolationMz"]
            msms_frame_extraction_succeeded = False
            for key in potential_tables_to_use:
                msms_info = data.table2dict(key)
                prec_mzs = None
                # check if all the necessary data is present.
                for prec_key in potential_msms_precursor_keys:
                    if prec_key in msms_info.keys():
                        prec_mzs = msms_info[prec_key]
                if prec_mzs is None:
                    continue
            
                msms_frames = msms_info['Frame']
                energies = msms_info['CollisionEnergy']
                msms_frame_extraction_succeeded = True
                break

            assert msms_frame_extraction_succeeded, "There was an issue obtaining the msms frame data."
            
            # Get list of all frames and their properties
            frame_ids = data.frames['Id']
            frame_prop = data.frame_properties

            # iterate through frames
            for frame_id in frame_ids:
                # Get mass and mobility range
                frame_info = data.query([frame_id],columns_for_check_dim)
                mass_range_start = round(min(frame_info['mz']),1)
                mass_range_end = round(max(frame_info['mz']),1)
                mob_range_start = round(min(frame_info['inv_ion_mobility']),3)
                mob_range_end = round(max(frame_info['inv_ion_mobility']),3)

                # Get time, polarity, and scan level
                rt = frame_prop[frame_id].Time
                polarity = frame_prop[frame_id].Polarity
                scanmode = frame_prop[frame_id].ScanMode
                level = parse_bruker_scan_level(scanmode)

                # Collect MS2 specific data if possible
                idx = np.where(msms_frames == frame_id)[0]

                if idx.shape[0]:
                    idx = idx[0]
                    mz = prec_mzs[idx]
                    energy = energies[idx]
                else:
                    mz = 0.0
                    energy = 0.0

                # save the filter with its corresponding retention time
                line_acq_times.append(rt)
                line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end, mob_range_start, mob_range_end])
                    
        acq_times.append(line_acq_times)
        filter_list.append(line_filter_list)

    num_spe_per_line = [len(i) for i in acq_times]
    # show results
    if ShowNumLineSpe:
        print('\nline scan spectra summary\n# of lines is: {}\nmean # of spectra is: {}\nmin # of spectra is: {}\nmean start time is {}\nmean end time is: {}'.format(
            len(num_spe_per_line), int(np.mean(num_spe_per_line)), int(np.min(num_spe_per_line)),np.mean([i[0] for i in acq_times]),np.mean([i[-1] for i in acq_times])))

    return acq_times, filter_list

def get_filters_info_d(all_filters_list):
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

def get_ScansPerFilter_d(line_list, filters_info, all_filters_list, filter_inverse, display_tqdm = False):
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

def get_filter_idx_d(Filter,acq_types,acq_polars,mz_ranges,precursors):

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

def reorder_pixels_d(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs):
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


# =====================================================================================
# Agilent
# =====================================================================================

class HiddenPrints:
    '''Allows code to be run without displaying messages.'''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# reference dicts for Agilent files
scanTypeDict = {7951 : "All",
                15 : "AllMS",
                7936 : "AllMSN",
                4 : "HighResolutionScan",
                256 : "MultipleReaction",
                4096 : "NeutralGain",
                2048 : "NeutralLoss",
                1024 : "PrecursorIon",
                512 : "ProductIon",
                1 : "Scan",
                2 : "SelectedIon",
                8 : "TotalIon",
                0 : "Unspecified"}
scanLevelDict = {0 : "All",
                1 : "ms",
                2 : "ms2"}
ionModeDict = {32 : "Apci",
            16 : "Appi",
            4 : "CI",
            2 : "EI",
            64 : "ESI",
            1024 : "ICP",
            2048 : "Jetstream",
            8 : "Maldi",
            1 : "Mixed",
            512 : "MsChip",
            128 : "NanoEsi",
            0 : "Unspecified"}
scanModeDict = {1 : "mixed", # Mixed
                3 : "centroid", # Peak
                2 : "profile", # Profile
                0 : "Unspecified"}
deviceTypeDict = {20 : "ALS",
                16 : "AnalogDigitalConverter",
                31 : "BinaryPump",
                31 : "CANValves",
                42 : "CapillaryPump",
                33 : "ChipCube",
                41 : "CTC",
                23 : "DiodeArrayDetector",
                14 : "ElectronCaptureDetector",
                17 : "EvaporativeLightScatteringDetector",
                19 : "FlameIonizationDetector",
                10 : "FlourescenceDetector",
                18 : "GCDetector",
                50 : "IonTrap",
                3 : "IsocraticPump",
                22 : "MicroWellPlateSampler",
                1 : "Mixed",
                13 : "MultiWavelengthDetector",
                34 : "Nanopump",
                2 : "Quadrupole",
                6 : "QTOF",
                32 : "QuaternaryPump",
                12 : "RefractiveIndexDetector",
                5 : "TandemQuadrupole",
                11 : "ThermalConductivityDetector",
                40 : "ThermostattedColumnCompartment",
                4 : "TOF",
                0 : "Unknown",
                15 : "VariableWavelengthDetector",
                21 : "WellPlateSampler"}
ionPolarityDict = {3 : "+-",
                1 : "-",
                0 : "+", 
                2 : "No Polarity"}
desiredModeDict = {'profile':0,
                'peak':1,
                'centroid':1,
                'profileelsepeak':2,
                'peakelseprofile':3}
    
def get_basic_instrument_metadata_agilent(line_list, data, metadata):
    
    metadata['file'] = line_list[0]
    metadata['Format'] = data.format
    metadata['AbundanceLimit'] = data.source.GetSpectrum(data.source, 1).AbundanceLimit
    metadata['Threshold'] = data.source.GetSpectrum(data.source, 1).Threshold

    metadata_vars = ['DeviceType', 'IonModes','MSLevel','ScanTypes','SpectraFormat']
    save_names = ['DeviceName', 'IonModes','MSLevel','ScanTypes','SpectraFormat']
    metadata_dicts = [deviceTypeDict,ionModeDict,scanLevelDict,scanTypeDict,scanModeDict]
    source = data.source.MSScanFileInformation
    metadata = msigen.get_attr_values(metadata, source, metadata_vars, save_names=save_names, metadata_dicts=metadata_dicts)

    if data.source.GetBPC(data.source).MeasuredMassRange:
        metadata['MassRange'] = list(data.source.GetBPC(data.source).MeasuredMassRange)

    return metadata

def get_agilent_scan(data, index):
    # faster implementation of multiplierz scan() method for agilent files
    mode = DesiredMSStorageType.ProfileElsePeak
    scanObj = data.source.GetSpectrum(data.source, index, data.noFilter, data.noFilter, mode)
    return np.array(scanObj.XArray), np.array(scanObj.YArray)

# =====================================================================================
# Agilent MS1 Workflow
# =====================================================================================

def agilent_d_ms1_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    # get mass windows
    MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = mass_lists
    lb, _, _, _, _, _, _ = lower_lims
    ub, _, _, _, _, _, _ = upper_lims

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    # initiate accumulator
    pixels = []
    rts = []

    for i, file_dir in tqdm(enumerate(line_list), total = len(line_list), desc='Progress through lines', disable = True): 
        
        with HiddenPrints():
            data = mzFile(file_dir)

        if i == 0:
            metadata = get_basic_instrument_metadata_agilent(line_list, data, metadata)

        # grab headers for all scans
        # the TIC here (headers[:,1]) is from centroided data. 
        # This means that it should not be used since default data extraction is in profile mode in MSIGen 
        headers = np.array(data.xic())
        
        assert len(headers)>0, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)
        assert headers.shape[1] == 2, 'Data from file {} is corrupt, not present, or not loading properly'.format(file_dir)
        
        line_rts = np.round(headers[:,0], 4)
        num_spe = len(line_rts)
        line_rts = np.array(line_rts[:num_spe])
        
        line_pixels = np.zeros((num_spe, MS1_list.shape[0]+1))

        for j in tqdm(range(num_spe), desc = 'Progress through line {}'.format(i+1), disable = (testing or gui)):
            # Update gui variables            
            if gui:
                tkinter_widgets[0]['value']=(100*i/len(line_list))+((100/len(line_list))*(j/num_spe))
                tkinter_widgets[0].update()
                tkinter_widgets[2]['text']=f'line {i+1}/{len(line_list)}, spectrum {j+1}/{num_spe}'
                tkinter_widgets[2].update()
            
            # get all intensity values for pixel 
            mz, intensity_points = get_agilent_scan(data, j)
            # remove all values of zero to improve speed
            intensity_points_mask = np.where(intensity_points)
            intensity_points = np.append(intensity_points[intensity_points_mask[0]],0)
            # get all m/z values with nonzero intensity
            mz = mz[intensity_points_mask[0]]

            # Get TIC
            line_pixels[j,0]=np.sum(intensity_points)
            
            if msigen.numba_present:
                idxs_to_sum = msigen.vectorized_sorted_slice_njit(mz, lb, ub)
                pixel = msigen.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                line_pixels[j,1:] = pixel
            else:
                idxs_to_sum = msigen.vectorized_sorted_slice(mz, lb, ub) # Slower
                line_pixels[j,1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

        data.close()

        pixels.append(line_pixels)
        rts.append(line_rts)

    metadata['average_start_time'] = np.mean([i[0] for i in rts])
    metadata['average_end_time'] = np.mean([i[-1] for i in rts])

    pixels_aligned = msigen.ms1_interp(pixels, rts, MS1_list, line_list)
    
    return metadata, pixels_aligned


# =====================================================================================
# Agilent MS2 Functions
# =====================================================================================

def agilent_d_ms2_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    # get mass windows
    MS1_list, _, MS1_polarity_list, prec_list, frag_list, _, MS2_polarity_list, mass_list_idxs = mass_lists

    acq_times, all_filters_list = check_dim_d(line_list, experiment_type)
    metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
    metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])
    
    # for MSMS, extracts info from filters
    filters_info, filter_inverse = get_filters_info_d(all_filters_list)
    # Determines correspondance of MZs to filters
    PeakCountsPerFilter, mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter \
        = msigen.get_PeakCountsPerFilter(filters_info, mass_lists, lower_lims, upper_lims)
    # finds the number of scans that use a specific filter
    scans_per_filter = get_ScansPerFilter_d(line_list, filters_info, all_filters_list, filter_inverse)
    # Groups filters into groups containing the same mzs/transitions
    consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
        = msigen.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)
    num_filter_groups = len(consolidated_filter_list)
    
    # get an array that gives the scan group number from the index of any scan (1d index)
    grp_from_scan_idx = np.empty((len(filters_info[0])), dtype = int)
    for idx, i in enumerate(consolidated_idx_list):
        for j in i:
            grp_from_scan_idx[j]=idx
    grp_from_scan_idx = grp_from_scan_idx[filter_inverse]

    all_TimeStamps = []
    pixels_metas = []

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    # holds index of current scan
    scan_idx = 0

    for i, Name in tqdm(enumerate(line_list), desc = 'Progress through lines', total = len(line_list), disable = (testing or gui)):                
        # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
        TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
        # counts how many times numbers have been inputted each array
        counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

        if Name.lower().endswith('.d'):
            with HiddenPrints():
                data = mzFile(Name)
            
            # collect metadata from raw file
            # if i == 0:
            #     metadata = get_basic_instrument_metadata_raw_no_mob(data, metadata)

            # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
            pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
            
            # old version kept in case of bugs 
            # TIC = get_TIC_agilent(data, acq_times[i])

            for j, TimeStamp in tqdm(enumerate(acq_times[i]), disable = True):
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
                TimeStamps[grp][counter[grp]] = TimeStamp 
                # get all intensity values for pixel 
                mz, intensity_points = get_agilent_scan(data, j)
                # Get TIC
                pixels_meta[grp][counter[grp], 0] = np.sum(intensity_points)

                # skip filters with no masses in the mass list
                if peak_counts_per_filter_grp[grp]:

                    # remove all values of zero to improve speed
                    intensity_points_mask = np.where(intensity_points)
                    intensity_points = np.append(intensity_points[intensity_points_mask[0]],0)
                    # get all m/z values with nonzero intensity
                    mz = mz[intensity_points_mask[0]]
                    
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

            data.close()

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
    pixels = reorder_pixels_d(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)    
    if normalize_img_sizes:
        pixels = msigen.pixels_list_to_array(pixels, line_list, all_TimeStamps_aligned)

    return metadata, pixels 

# ================================================================================
# Bruker General functions
# ================================================================================

def parse_bruker_scan_level(scanmode):
    if scanmode == 0:
        level = 'MS1'
    elif scanmode == 2:
        level = 'MRM'
    elif scanmode == 8:
        level = 'ddaPASEF'
    elif scanmode == 9:
        level = 'diaPASEF'
    else:
        level = 'MS1'
        raise RuntimeWarning('The file contains scan types that are not MS1, MRM, ddaPASEF, or diaPASEF.\nBehavior may differ from expected behavior')
    return level


# ================================================================================
# Bruker tsf MS1
# ================================================================================

def tsf_d_ms1_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    # get mass windows
    MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = mass_lists
    lb, _, _, _, _, _, _ = lower_lims
    ub, _, _, _, _, _, _ = upper_lims
    
    # initiate accumulator
    pixels = []
    rts = []

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()


    for i, file_dir in tqdm(enumerate(line_list), total = len(line_list), desc='Progress through lines', disable = True): 
        data = tsf.tsf_data(file_dir, tsf.dll)
        
        # get line tics and acquisiton times
        TICs = data.frames['SummedIntensities'].values
        line_rts = data.frames['Time'].values 
        num_spe = TICs.shape[0]

        # Initialize line collector
        line_pixels = np.zeros((num_spe, MS1_list.shape[0]+1))
        line_pixels[:,0] = TICs

        for j in tqdm(range(1,num_spe+1), desc = 'Progress through line {}'.format(i+1), disable = (testing or gui)):
            # Update gui variables            
            if gui:
                tkinter_widgets[0]['value']=(100*i/len(line_list))+((100/len(line_list))*(j/num_spe))
                tkinter_widgets[0].update()
                tkinter_widgets[2]['text']=f'line {i+1}/{len(line_list)}, spectrum {j+1}/{num_spe}'
                tkinter_widgets[2].update()

            # get profile spectrum
            intensity_points = data.read_profile_spectrum(j)
            # remove zero values to reduce the number of mz values to retrieve
            intensity_points_mask = np.where(intensity_points)
            intensity_points = intensity_points[intensity_points_mask]
            # retrieve mz values of nonzero mz valyes
            mz = data.index_to_mz(j,intensity_points_mask[0])

            if msigen.numba_present:
                idxs_to_sum = msigen.vectorized_sorted_slice_njit(mz, lb, ub)

                pixel = msigen.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                line_pixels[j-1,1:] = pixel
            else:
                idxs_to_sum = msigen.vectorized_sorted_slice(mz, lb, ub) # Slower
                line_pixels[j-1,1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

        del data

        pixels.append(line_pixels)
        rts.append(line_rts)

    metadata['average_start_time'] = np.mean([i[0] for i in rts])
    metadata['average_end_time'] = np.mean([i[-1] for i in rts])

    pixels_aligned = msigen.ms1_interp(pixels, rts, MS1_list, line_list)

    return metadata, pixels_aligned

def get_basic_instrument_metadata_bruker_d_no_mob(line_list, data, metadata = {}):
    ### I need to make sure the dict keys line up between the different instruments
    metadata['format'] = data.metadata['AcquisitionSoftwareVendor']+'-'+data.metadata['SchemaType']
    metadata['file'] = line_list[0]
    metadata['InstrumentVendor'] = data.metadata['InstrumentVendor']
    metadata['SchemaType'] = data.metadata['SchemaType']
    metadata['SchemaVersionMajor'] = data.metadata['SchemaVersionMajor']
    metadata['SchemaVersionMinor'] = data.metadata['SchemaVersionMinor']
    metadata['TimsCompressionType'] = data.metadata['TimsCompressionType']
    metadata['AcquisitionSoftware'] = data.metadata['AcquisitionSoftware']
    metadata['AcquisitionSoftwareVendor'] = data.metadata['AcquisitionSoftwareVendor']
    metadata['AcquisitionSoftwareVersion'] = data.metadata['AcquisitionSoftwareVersion']
    metadata['AcquisitionFirmwareVersion'] = data.metadata['AcquisitionFirmwareVersion']
    metadata['InstrumentName'] = data.metadata['InstrumentName']
    metadata['InstrumentFamily'] = data.metadata['InstrumentFamily']
    metadata['InstrumentRevision'] = data.metadata['InstrumentRevision']
    metadata['InstrumentSourceType'] = data.metadata['InstrumentSourceType']
    metadata['InstrumentSerialNumber'] = data.metadata['InstrumentSerialNumber']
    metadata['Description'] = data.metadata['Description']
    metadata['SampleName'] = data.metadata['SampleName']
    metadata['MethodName'] = data.metadata['MethodName']
    metadata['DenoisingEnabled'] = data.metadata['DenoisingEnabled']
    metadata['PeakWidthEstimateValue'] = data.metadata['PeakWidthEstimateValue']
    metadata['PeakWidthEstimateType'] = data.metadata['PeakWidthEstimateType']
    metadata['HasProfileSpectra'] = data.metadata['HasProfileSpectra']
    
    return metadata

# ================================================================================
# tsf MS2
# ================================================================================

def tsf_d_ms2_no_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    # get mass windows
    MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = mass_lists
    
    acq_times, all_filters_list = check_dim_d(line_list, experiment_type)
    metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
    metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

    # for MSMS, extracts info from filters
    filters_info, filter_inverse = get_filters_info_d(all_filters_list)
    # Determines correspondance of MZs to filters
    PeakCountsPerFilter, mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter \
        = msigen.get_PeakCountsPerFilter(filters_info, mass_lists, lower_lims, upper_lims)
    # finds the number of scans that use a specific filter
    scans_per_filter = get_ScansPerFilter_d(line_list, filters_info, all_filters_list, filter_inverse)
    # Groups filters into groups containing the same mzs/transitions
    consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
        = msigen.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)
    num_filter_groups = len(consolidated_filter_list)

    # get an array that gives the scan group number from the index of any scan (1d index)
    grp_from_scan_idx = np.empty((len(filters_info[0])), dtype = int)
    for idx, i in enumerate(consolidated_idx_list):
        for j in i:
            grp_from_scan_idx[j]=idx
    grp_from_scan_idx = grp_from_scan_idx[filter_inverse]

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

        if Name.lower().endswith('.d'):
            data = tsf.tsf_data(Name, tsf.dll)
            
            # collect metadata from raw file
            # if i == 0:
            #     metadata = get_basic_instrument_metadata_raw_no_mob(data, metadata)

            # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
            pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
            
            frames_info = data.frames
            TIC = frames_info["SummedIntensities"].values

            for j, TimeStamp in tqdm(enumerate(acq_times[i]), disable = True):
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
                TimeStamps[grp][counter[grp]] = TimeStamp 
                pixels_meta[grp][counter[grp], 0] = TIC[j]

                # skip filters with no masses in the mass list
                if peak_counts_per_filter_grp[grp]:
                    
                    # get spectrum
                    intensity_points = data.read_profile_spectrum(j+1)
                    # remove all values of zero to improve speed
                    intensity_points_mask = np.where(intensity_points)[0]
                    intensity_points = intensity_points[intensity_points_mask]
                    # get all m/z values with nonzero intensity
                    mz = data.index_to_mz(j+1, intensity_points_mask)

                    lbs,ubs = np.array(mzs_per_filter_grp_lb[grp], dtype = np.float64), np.array(mzs_per_filter_grp_ub[grp], dtype = np.float64)
                
                    if msigen.numba_present:
                        idxs_to_sum = msigen.vectorized_sorted_slice_njit(mz, lbs, ubs)
                        pixel = msigen.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                        pixels_meta[grp][counter[grp],1:] = pixel
                    else:
                        idxs_to_sum = msigen.vectorized_sorted_slice(mz, lbs, ubs) # Slower
                        pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

                # keep count of the 1d scan index
                scan_idx += 1

            del data

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
    pixels = reorder_pixels_d(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)    
    if normalize_img_sizes:
        pixels = msigen.pixels_list_to_array(pixels, line_list, all_TimeStamps_aligned)


    return metadata, pixels


# ================================================================================
# tsf MS1
# ================================================================================

def tdf_d_ms1_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()
    
    # get mass windows
    MS1_list, MS1_mob_list, MS1_polarity_list, _, _, _, _, mass_list_idxs = mass_lists
    lb, mob_lb, _, _, _, _, _ = lower_lims
    ub, mob_ub, _, _, _, _, _ = upper_lims

    # determine range of mz and mob values to consider
    min_mz, max_mz = np.min(lb), np.max(ub)
    min_mob, max_mob = np.min(mob_lb), np.max(mob_ub)
    
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    # initiate accumulator
    pixels = []
    rts = []

    # iterate over lines
    for i, file_dir in tqdm(enumerate(line_list), total = len(line_list), desc='Progress through lines', disable = True): 
        
        # open line data file
        data = OpenTIMS(file_dir)

        # get general data to initialize the line pixels array
        line_rts = data.retention_times
        num_spe = len(line_rts)
        line_pixels = np.zeros((num_spe, len(MS1_list)+1))

        # iterate over spectrum iterator
        for j, frame_dict in tqdm(enumerate(data.query_iter(frames = data.frames['Id'], columns = ('mz', 'inv_ion_mobility', 'intensity'))), \
                                "progress thru line {linenum} of {total_lines}".format(linenum = i+1, total_lines = len(line_list)), total = num_spe, delay = 0.05, disable = (testing or gui)):
            # Update gui variables            
            if gui:
                tkinter_widgets[0]['value']=(100*i/len(line_list))+((100/len(line_list))*(j/num_spe))
                tkinter_widgets[0].update()
                tkinter_widgets[2]['text']=f'line {i+1}/{len(line_list)}, spectrum {j+1}/{num_spe}'
                tkinter_widgets[2].update()

            # get all peaks
            mz = frame_dict['mz']
            mob = frame_dict['inv_ion_mobility']
            intensity_points = frame_dict['intensity']

            #TIC
            line_pixels[j,0] = np.sum(intensity_points)

            # remove peaks that are not possibly in the mass range of given mass list to improve speed
            mask = np.where((mz>min_mz)&(mz<max_mz)&(mob>min_mob)&(mob<max_mob))
            mz, mob, intensity_points = mz[mask],mob[mask],intensity_points[mask]

            # find peaks with mass and mobility window and sum them
            for k in range(len(MS1_list)):
                
                mask = np.where((mz>lb[k])&(mz<ub[k])&(mob<mob_ub[k])&(mob>mob_lb[k]))
                line_pixels[j,k+1] = np.sum(intensity_points[mask])

        pixels.append(line_pixels)
        rts.append(line_rts)

    pixels_aligned = msigen.ms1_interp(pixels, rts, MS1_list, line_list)

    return metadata, pixels_aligned 

def tdf_d_ms2_mob(line_list, mass_lists, lower_lims, upper_lims, experiment_type, metadata, normalize_img_sizes = True, in_jupyter = True, testing = False, gui=False, tkinter_widgets = [None, None, None]):
    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Preprocessing data"
        tkinter_widgets[1].update()

    # get mass windows
    MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = mass_lists

    acq_times, all_filters_list = check_dim_d(line_list, experiment_type = 0, ShowNumLineSpe=in_jupyter)

    metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
    metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

    filters_info, filter_inverse = get_filters_info_tdf(all_filters_list)

    PeakCountsPerFilter, mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mobsPerFilter_lb, mobsPerFilter_ub, mzIndicesPerFilter \
        = msigen.get_PeakCountsPerFilter(filters_info, mass_lists, lower_lims, upper_lims)

    scans_per_filter = get_ScansPerFilter_d(line_list, filters_info, all_filters_list, filter_inverse)

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

    all_TimeStamps = []
    pixels_metas = []

    # keywords used in extracting data later
    columns_for_data_extraction = ('mz','inv_ion_mobility','intensity')

    # variables for monitoring progress on gui
    if gui:
        tkinter_widgets[1]['text']="Extracting data"
        tkinter_widgets[1].update()

    # holds index of current scan/spectrum
    scan_idx = 0

    for i, file_dir in tqdm(enumerate(line_list), desc = 'Progress through lines', total = len(line_list), disable = (testing or gui)):                
        # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
        TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
        # counts how many times numbers have been inputted each array
        counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

        data = OpenTIMS(file_dir)
        
        # collect metadata from raw file
        # if i == 0:
        #     metadata = get_basic_instrument_metadata_raw_no_mob(data, metadata)

        # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
        pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
        
        frames_info = data.frames
        TIC = data.framesTIC()

        for j, TimeStamp in tqdm(enumerate(acq_times[i]), disable = True):
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
            TimeStamps[grp][counter[grp]] = TimeStamp 
            pixels_meta[grp][counter[grp], 0] = TIC[j]

            # skip filters with no masses in the mass list
            if peak_counts_per_filter_grp[grp]:
                frame_info = data.query([j+1],columns_for_data_extraction)
                mz = frame_info['mz']
                mob = frame_info['inv_ion_mobility']
                intensity_points = np.append(frame_info['intensity'],0)      
                
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

        data.close()
        del data

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
    # pixels = (pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs, line_list)    
    pixels = reorder_pixels_d(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)    
    if normalize_img_sizes:
        pixels = msigen.pixels_list_to_array(pixels, line_list, all_TimeStamps_aligned)

    return metadata, pixels

def get_filters_info_tdf(all_filters_list):
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




