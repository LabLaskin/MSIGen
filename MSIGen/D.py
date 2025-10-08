"""
This module contains a subclass of the base MSIGen class for handling files with the .d file extension.
This includes Bruker .tsf, .baf, and .tdf formats and Agilent formats that do not contain ion mobility data.
"""

from MSIGen.base_class import MSIGen_base

import os, sys
import numpy as np
from tqdm import tqdm
from MSIGen.base_class import custom_warning
try:
    from MSIGen import tsf
    assert "dll" in dir(tsf)
except (ImportError, AssertionError):
    custom_warning("Cannot extract Bruker .tsf data. Check that you input the timsdata.dll in the MSIGen package folder. \n \
          This can be safely ignored if you are not using Bruker .tsf files.")

try:
    from pyBaf2Sql.init_baf2sql import init_baf2sql_api
    from pyBaf2Sql.classes import BafData
    from pyBaf2Sql.baf import read_double
except ImportError:
    custom_warning("Could not import pyBaf2Sql. Cannot extract Bruker .baf data. Check that pyBaf2Sql is installed.\n \
          Can be found at https://github.com/gtluu/pyBaf2Sql \n \
          This can be safely ignored if you are not using Bruker .baf files.")

# bruker tdf
try:
    from opentimspy.opentims import OpenTIMS
except ImportError:
    custom_warning('Could not import openTIMSpy. Cannot process .tdf format data from Bruker TIMS-TOF\n \
          This can be safely ignored if you are not using Bruker .tdf files.')

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

except Exception as e: 
    custom_warning("Could not import mzFile or associated dll's. Cannot process Agilent's .d format data \n \
          This can be safely ignored if you are not using Agilent .d files.", err=e)

# =====================================================================================
# General functions
# =====================================================================================
class MSIGen_D(MSIGen_base):
    """
    MSIGen_D class for processing mass spectrometry data from Agilent and Bruker formats.

    Inherits from the base MSIGen_base class and implements methods for loading and processing data.

    This class is designed to handle different file formats, including Agilent .d files and Bruker .tsf/.baf/.tdf files.

    Attributes:
        data_format (str, None): 
            The format of the data file. Can be "bruker_tsf", "bruker_baf", "bruker_tdf", or "agilent".
        scanTypeDict (dict):
            Dictionary mapping scan types to their descriptions for Agilent files.
        scanLevelDict (dict):
            Dictionary mapping scan levels to their descriptions for Agilent files.
        ionModeDict (dict):
            Dictionary mapping ion modes to their descriptions for Agilent files.
        scanModeDict (dict):
            Dictionary mapping scan modes to their descriptions for Agilent files.
        deviceTypeDict (dict):
            Dictionary mapping device types to their descriptions for Agilent files.
        ionPolarityDict (dict):
            Dictionary mapping ion polarities to their descriptions for Agilent files.
        desiredModeDict (dict):
            Dictionary mapping desired modes to their corresponding values for Agilent files.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data_format, _ = self.determine_file_format(self.example_file) # "bruker_tsf", "bruker_tdf", or "Agilent"
        
        # reference dicts for Agilent files
        self.scanTypeDict = {7951 : "All",
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
        self.scanLevelDict = {0 : "All",
                        1 : "ms",
                        2 : "ms2"}
        self.ionModeDict = {32 : "Apci",
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
        self.scanModeDict = {1 : "mixed", # Mixed
                        3 : "centroid", # Peak
                        2 : "profile", # Profile
                        0 : "Unspecified"}
        self.deviceTypeDict = {20 : "ALS",
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
        self.ionPolarityDict = {3 : "+-",
                        1 : "-",
                        0 : "+", 
                        2 : "No Polarity"}
        self.desiredModeDict = {'profile':0,
                        'peak':1,
                        'centroid':1,
                        'profileelsepeak':2,
                        'peakelseprofile':3}

    def load_files(self, *args, **kwargs):
        """Processes the data files based on the specified data format, MS level, and whether ion mobility data are present."""
        if self.data_format.lower() == "agilent":
            if (not self.is_MS2) and (not self.is_mobility):
                return self.agilent_d_ms1_no_mob(*args, **kwargs)
            elif (self.is_MS2) and (not self.is_mobility):
                return self.agilent_d_ms2_no_mob(*args, **kwargs)
            else:
                raise NotImplementedError('Mobility data not yet supported for Agilent .d files.\nConvert the files to mzML before using MSIGen for processing Agilent files containing ion mobility data.')
        elif self.data_format.lower() in ["bruker_tsf", "bruker_baf"]:
            if (not self.is_MS2) and (not self.is_mobility):
                return self.bruker_d_ms1_no_mob(*args, **kwargs)
            elif (self.is_MS2) and (not self.is_mobility):
                return self.bruker_d_ms2_no_mob(*args, **kwargs)
            else:
                raise NotImplementedError('Mobility data is not supported for {} files.'.format(self.data_format))
        elif self.data_format.lower() == "bruker_tdf":
            if (not self.is_MS2) and (self.is_mobility):
                return self.tdf_d_ms1_mob(*args, **kwargs)
            elif (self.is_MS2) and (self.is_mobility):
                return self.tdf_d_ms2_mob(*args, **kwargs)
            else:
                raise NotImplementedError('Bruker .tdf data must contain ion mobility data.')
        else:
            raise NotImplementedError("The data format was not able to be recognized")

    def determine_file_format(self, example_file=None):
        """
        Determines the file format and MS level of the provided example file.
        If the data contains any MS2 scans, the MS level is "MS2", otherwise it is "MS1".

        Returns:
            data_format (str): The format of the data file. Can be "agilent", "bruker_tsf", "bruker_baf", or "bruker_tdf".
            MS_level (str): The MS level of the data file. Can be "MS1" or "MS2".
        """
        if example_file:
            setattr(self, "example_file", example_file)

        if type(example_file) in [list, tuple]:
            f = example_file[0]
        elif type(example_file) == str:
            f = example_file
        else:
            raise ValueError("The example file must be a string or a list of strings.")
        
        MS_level = 'Not specified'
        if os.path.exists(os.path.join(f, 'analysis.tsf')):
            data_format = "bruker_tsf"
        elif os.path.exists(os.path.join(f, 'analysis.tdf')):
            data_format = "bruker_tdf"
        elif os.path.exists(os.path.join(f, 'analysis.baf')):
            data_format = "bruker_baf"
        else:
            try:
                data = mzFile(f)
                # vendor format
                data_format = data.format #(Almost definitely "Agilent")
                # MS1 or MS2
                MS_levels = np.unique(np.array(data.scan_info())[:,3])
                if 'MS2' in MS_levels:
                    MS_level = 'MS2'
                else:
                    MS_level = 'MS1'
                data.close()
            except Exception as e:
                raise RuntimeError("Data file could not be read:\n{}".format(e))

        return data_format, MS_level
    
    # TODO: Add this method for other formats, especially tdf and baf.
    def get_basic_instrument_metadata(self, data, metadata=None):
        """Gets some of the instrument metadata from the data file depending on the file format."""
        if self.data_format.lower() == "agilent":
            self.metadata = self.get_basic_instrument_metadata_agilent(data, self.metadata)
        elif self.data_format.lower() == "bruker_tsf":
            self.metadata = self.get_basic_instrument_metadata_bruker_d_tsf_no_mob(data, self.metadata)
        else:
            custom_warning("The method for obtaining metadata for this file format is not implemented yet.")
        return self.metadata
    
    def check_dim(self, ShowNumLineSpe=False):
        """
        Gets the acquisition times and other information about each scan to 
        decide what mass list entries can be obtained from each scan.
        
        Returns:
            acq_times (list): A list of acquisition times for each line.
            filter_list (list): A list of information that would be included in Thermo-style filter strings for each line.
        """
        acq_times = []
        filter_list = []

        for file_dir in self.line_list:
            line_acq_times = []
            line_filter_list = []

            if self.data_format.lower() == 'agilent':
                # Get Start times, end times, number of spectra per line, and list of unique filters.
                with self.HiddenPrints():
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
            
            elif self.data_format.lower() == 'bruker_tsf':
                data = tsf.tsf_data(file_dir, tsf.dll)
                
                msms_info = data.framemsmsinfo
                frames_info = data.frames.values
                for idx, rt, polarity, scanmode, msms_type, _, max_intensity, tic, num_peaks, MzCalibration, _,_,_ in frames_info:
                    
                    # get the values necessary for filters 
                    [[mz, energy]] = msms_info[msms_info["Frame"]==1][['TriggerMass', 'CollisionEnergy']].values
                    mass_range_start, mass_range_end = data.index_to_mz(idx, [0, len(data.read_profile_spectrum(idx))])
                    level = self.parse_bruker_scan_level(scanmode)
                    # save those values
                    line_acq_times.append(rt)
                    line_filter_list.append([mz, energy, level, polarity, mass_range_start, mass_range_end])
                del data

            elif self.data_format.lower() == 'bruker_baf':
                raise NotImplementedError('MS2 data in Bruker .baf format is not supported yet.')
                data = BafData(bruker_d_folder_name=file_dir, baf2sql=init_baf2sql_api())
                # get line tics and acquisiton times
                TICs = data.analysis["Spectra"]["SumIntensity"].values
                line_rts = data.analysis["Spectra"]["Rt"].values


            elif self.data_format.lower() == 'bruker_tdf':
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
                    level = self.parse_bruker_scan_level(scanmode)

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

    def get_ScansPerFilter(self, filters_info, all_filters_list, filter_inverse, display_tqdm = False):
        """Determines the number of scans that use a specific filter"""
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

    def get_filter_idx(self,Filter,acq_types,acq_polars,mz_ranges,precursors):
        """
        Gets the index of the filter that corresponds to the given filter information.
        This is unused in the current implementation.
        """

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

    # =====================================================================================
    # Agilent
    # =====================================================================================
    def get_basic_instrument_metadata_agilent(self, data, metadata=None):
        """Obtains basic instrument metadata from Agilent data."""
        try:
            self.metadata['file'] = self.line_list[0]
            self.metadata['Format'] = data.format
            self.metadata['AbundanceLimit'] = data.source.GetSpectrum(data.source, 1).AbundanceLimit
            self.metadata['Threshold'] = data.source.GetSpectrum(data.source, 1).Threshold

            metadata_vars = ['DeviceType', 'IonModes','MSLevel','ScanTypes','SpectraFormat']
            save_names = ['DeviceName', 'IonModes','MSLevel','ScanTypes','SpectraFormat']
            metadata_dicts = [self.deviceTypeDict,self.ionModeDict,self.scanLevelDict,self.scanTypeDict,self.scanModeDict]
            source = data.source.MSScanFileInformation
            self.metadata = self.get_attr_values(self.metadata, source, metadata_vars, save_names=save_names, metadata_dicts=metadata_dicts)

            if data.source.GetBPC(data.source).MeasuredMassRange:
                self.metadata['MassRange'] = list(data.source.GetBPC(data.source).MeasuredMassRange)
        except Exception as e:
            custom_warning("There was an issue obtaining some of the metadata from the Agilent file.", err=e)
        return self.metadata
    
    @staticmethod
    def get_agilent_scan(data, index):
        """
        A faster implementation of the scan() method from multiplierz's mzFile package for Agilent files.
        
        Returns:
            mz (np.ndarray): The m/z values of the scan.
            intensity_points (np.ndarray): The intensity values of the scan.
        """
        mode = DesiredMSStorageType.ProfileElsePeak
        scanObj = data.source.GetSpectrum(data.source, index, data.noFilter, data.noFilter, mode)
        return np.array(scanObj.XArray), np.array(scanObj.YArray)

    # =====================================================================================
    # Agilent MS1 Workflow
    # =====================================================================================
    def agilent_d_ms1_no_mob(self, metadata=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None):
        """
        Data processing for Agilent .d files with only MS1 data.
        
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

        for i, file_dir in tqdm(enumerate(self.line_list), total = len(self.line_list), desc='Progress through lines', disable = True): 
            
            with self.HiddenPrints():
                data = mzFile(file_dir)

            if i == 0:
                # updates the metadata dictionary with instrument information
                self.metadata = self.get_basic_instrument_metadata(data, self.metadata)

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

            for j in tqdm(range(num_spe), desc = 'Progress through line {}'.format(i+1), disable = (self.testing or self.gui)):
                # Update gui variables
                self.progressbar_update_progress(num_spe, i, j)
                
                # get all intensity values for pixel 
                mz, intensity_points = self.get_agilent_scan(data, j)
                # remove all values of zero to improve speed
                intensity_points_mask = np.where(intensity_points)
                intensity_points = np.append(intensity_points[intensity_points_mask[0]],0)
                # get all m/z values with nonzero intensity
                mz = mz[intensity_points_mask[0]]

                # Get TIC
                line_pixels[j,0]=np.sum(intensity_points)
                
                pixel = self.extract_masses_no_mob(mz, lb, ub, intensity_points)
                line_pixels[j,1:] = pixel

                # if self.numba_present:
                #     idxs_to_sum = self.vectorized_sorted_slice_njit(mz, lb, ub)
                #     pixel = self.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                #     line_pixels[j,1:] = pixel
                # else:
                #     idxs_to_sum = self.vectorized_sorted_slice(mz, lb, ub) # Slower
                #     line_pixels[j,1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

            data.close()

            pixels.append(line_pixels)
            rts.append(line_rts)

        self.metadata['average_start_time'] = np.mean([i[0] for i in rts])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in rts])

        self.rts = rts
        pixels_aligned = self.ms1_interp(pixels, mass_list = MS1_list)
        
        return self.metadata, pixels_aligned


    # =====================================================================================
    # Agilent MS2 Functions
    # =====================================================================================

    def agilent_d_ms2_no_mob(self, metadata=None, normalize_img_sizes=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None):
        """
        Data processing for Agilent .d files that contain MS2 data.
        
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
        for i in [("normalize_img_sizes", normalize_img_sizes), ("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets), ("metadata", metadata)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])

        # monitor progress on gui
        self.progressbar_start_preprocessing()

        # get mass windows
        MS1_list, _, MS1_polarity_list, prec_list, frag_list, _, MS2_polarity_list, mass_list_idxs = self.mass_list

        acq_times, all_filters_list = self.check_dim()
        self.metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])
        
        # for MSMS, extracts info from filters
        filters_info, filter_inverse = self.get_filters_info(all_filters_list)
        # Determines correspondance of MZs to filters
        mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter \
            = self.get_CountsPerFilter(filters_info)
        # finds the number of scans that use a specific filter
        scans_per_filter = self.get_ScansPerFilter(filters_info, all_filters_list, filter_inverse)
        # Groups filters into groups containing the same mzs/transitions
        consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, \
            mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
            = self.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)
        num_filter_groups = len(consolidated_filter_list)
        
        # get an array that gives the scan group number from the index of any scan (1d index)
        grp_from_scan_idx = np.empty((len(filters_info[0])), dtype = int)
        for idx, i in enumerate(consolidated_idx_list):
            for j in i:
                grp_from_scan_idx[j]=idx
        grp_from_scan_idx = grp_from_scan_idx[filter_inverse]

        all_TimeStamps = []
        pixels_metas = []

        # monitor progress on gui
        self.progressbar_start_extraction()

        # holds index of current scan
        scan_idx = 0

        for i, Name in tqdm(enumerate(self.line_list), desc = 'Progress through lines', total = len(self.line_list), disable = (self.testing or self.gui)):                
            # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
            TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
            # counts how many times numbers have been inputted each array
            counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

            with self.HiddenPrints():
                data = mzFile(Name)
            
            # collect metadata from raw file
            if i == 0:
                self.metadata = self.get_basic_instrument_metadata(data, self.metadata)

            # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
            pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
            
            # old version kept in case of bugs 
            # TIC = get_TIC_agilent(data, acq_times[i])

            for j, TimeStamp in tqdm(enumerate(acq_times[i]), disable = True):
                # Update gui variables
                self.progressbar_update_progress(len(acq_times[i]), i, j)

                # determine which group is going to be used
                grp = grp_from_scan_idx[scan_idx]
                counter[grp]+=1

                # handle info
                TimeStamps[grp][counter[grp]] = TimeStamp 
                # get all intensity values for pixel 
                mz, intensity_points = self.get_agilent_scan(data, j)
                # Get TIC
                pixels_meta[grp][counter[grp], 0] = np.sum(intensity_points)

                # skip filters with no masses in the mass list
                if peak_counts_per_filter_grp[grp]:

                    # remove all values of zero to improve speed
                    intensity_points_mask = np.where(intensity_points)
                    intensity_points = np.append(intensity_points[intensity_points_mask[0]],0)
                    # get all m/z values with nonzero intensity
                    mz = mz[intensity_points_mask[0]]
                    
                    lb,ub = np.array(mzs_per_filter_grp_lb[grp]), np.array(mzs_per_filter_grp_ub[grp])
                
                    pixel = self.extract_masses_no_mob(mz, lb, ub, intensity_points)
                    pixels_meta[grp][counter[grp],1:] = pixel

                    ## removed because of errors or slowness
                    # if self.numba_present:
                    #     idxs_to_sum = self.vectorized_sorted_slice_njit(mz, lb, ub)
                    #     pixel = self.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                    #     pixels_meta[grp][counter[grp],1:] = pixel
                    # else:
                    #     idxs_to_sum = self.vectorized_sorted_slice(mz, lbs, ubs) # Slower
                    #     pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

                # keep count of the 1d scan index
                scan_idx += 1

            data.close()

            all_TimeStamps.append(TimeStamps)
            pixels_metas.append(pixels_meta)

        self.rts = acq_times
        pixels, all_TimeStamps_aligned = self.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, mzs_per_filter_grp)

        # Order the pixels in the way the mass list csv/excel file was ordered
        pixels = self.reorder_pixels(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)    
        if normalize_img_sizes:
            pixels = self.pixels_list_to_array(pixels, all_TimeStamps_aligned)

        return self.metadata, pixels 

    # ================================================================================
    # Bruker General functions
    # ================================================================================
    @staticmethod
    def parse_bruker_scan_level(scanmode):
        """Obtains a descriptive scan mode string from the scan mode integer of a Bruker scan."""
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
    # Bruker tsf/baf MS1
    # ================================================================================

    def bruker_d_ms1_no_mob(self, metadata=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None):
        """
        Data processing from Bruker .tsf/.baf files containing only MS1 data.
        
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
        for i in [("in_jupyter", in_jupyter), ("testing", testing), ("gui", gui), ("pixels_per_line", pixels_per_line), ("tkinter_widgets", tkinter_widgets), ("metadata", metadata)]:
            if i[1] is not None:
                setattr(self, i[0], i[1])

        # monitor progress on gui
        self.progressbar_start_preprocessing()

        # get mass windows
        MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = self.mass_list
        lb, _, _, _, _, _, _ = self.lower_lims
        ub, _, _, _, _, _, _ = self.upper_lims
        
        # initiate accumulator
        pixels = []
        rts = []

        # variables for monitoring progress on gui
        if self.gui:
            tkinter_widgets[1]['text']="Extracting data"
            tkinter_widgets[1].update()


        for i, file_dir in tqdm(enumerate(self.line_list), total = len(self.line_list), desc='Progress through lines', disable = True): 
            if self.data_format.lower() == 'bruker_tsf':
                data = tsf.tsf_data(file_dir, tsf.dll)
                # get line tics and acquisiton times
                TICs = data.frames['SummedIntensities'].values
                line_rts = data.frames['Time'].values 

            elif self.data_format.lower() == 'bruker_baf':
                data = BafData(bruker_d_folder_name=file_dir, baf2sql=init_baf2sql_api())
                # get line tics and acquisiton times
                TICs = data.analysis["Spectra"]["SumIntensity"].values
                line_rts = data.analysis["Spectra"]["Rt"].values

            if i == 0:
                # updates the metadata dictionary with instrument information
                self.metadata = self.get_basic_instrument_metadata(data, self.metadata)

            num_spe = TICs.shape[0]

            # Initialize line collector
            line_pixels = np.zeros((num_spe, MS1_list.shape[0]+1))
            line_pixels[:,0] = TICs

            for j in tqdm(range(1,num_spe+1), desc = 'Progress through line {}'.format(i+1), disable = (self.testing or self.gui)):
                # Update gui variables
                self.progressbar_update_progress(num_spe, i, j)

                # get profile spectrum
                if self.data_format.lower() == 'bruker_tsf':
                    intensity_points = data.read_profile_spectrum(j)
                    # remove zero values to reduce the number of mz values to retrieve
                    intensity_points_mask = np.where(intensity_points)
                    intensity_points = intensity_points[intensity_points_mask]
                    # retrieve mz values of nonzero mz valyes
                    mz = data.index_to_mz(j,intensity_points_mask[0])

                if self.data_format.lower() == 'bruker_baf':
                    frame_dict = data.analysis['Spectra'][data.analysis['Spectra']['Id'] == j].to_dict(orient='records')[0]
                    intensity_points = np.array(read_double(baf2sql=init_baf2sql_api(), handle=data.handle, identity=frame_dict['ProfileIntensityId']),
                                                dtype=np.float64)
                    # remove zero values to reduce the number of mz values to retrieve
                    intensity_points_mask = np.where(intensity_points)
                    intensity_points = intensity_points[intensity_points_mask]
                    # retrieve mz values of nonzero mz valyes
                    mz = np.array(read_double(baf2sql=init_baf2sql_api(), handle=data.handle, identity=int(frame_dict['ProfileMzId'])),
                                            dtype=np.float64)[intensity_points_mask]
                    
                pixel = self.extract_masses_no_mob(mz, lb, ub, intensity_points)
                line_pixels[j-1,1:] = pixel

                # if self.numba_present:
                #     idxs_to_sum = self.vectorized_sorted_slice_njit(mz, lb, ub)
                #     pixel = self.assign_values_to_pixel_njit(intensity_points, idxs_to_sum)
                #     line_pixels[j-1,1:] = pixel
                # else:
                #     idxs_to_sum = self.vectorized_sorted_slice(mz, lb, ub) # Slower
                #     line_pixels[j-1,1:] = np.sum(np.take(intensity_points, idxs_to_sum), axis = 1)

            del data

            pixels.append(line_pixels)
            rts.append(line_rts)

        self.metadata['average_start_time'] = np.mean([i[0] for i in rts])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in rts])

        self.rts = rts
        pixels_aligned = self.ms1_interp(pixels, mass_list = MS1_list)

        return self.metadata, pixels_aligned

    def get_basic_instrument_metadata_bruker_d_tsf_no_mob(self, data, metadata = {}):
        """Obtains basic instrument metadata from Bruker .tsf data."""
        #TODO: I need to make sure the dict keys line up between the different instruments
        try:
            self.metadata['format'] = data.metadata['AcquisitionSoftwareVendor']+'-'+data.metadata['SchemaType']
            self.metadata['file'] = self.line_list[0]
            self.metadata['InstrumentVendor'] = data.metadata['InstrumentVendor']
            self.metadata['SchemaType'] = data.metadata['SchemaType']
            self.metadata['SchemaVersionMajor'] = data.metadata['SchemaVersionMajor']
            self.metadata['SchemaVersionMinor'] = data.metadata['SchemaVersionMinor']
            self.metadata['TimsCompressionType'] = data.metadata['TimsCompressionType']
            self.metadata['AcquisitionSoftware'] = data.metadata['AcquisitionSoftware']
            self.metadata['AcquisitionSoftwareVendor'] = data.metadata['AcquisitionSoftwareVendor']
            self.metadata['AcquisitionSoftwareVersion'] = data.metadata['AcquisitionSoftwareVersion']
            self.metadata['AcquisitionFirmwareVersion'] = data.metadata['AcquisitionFirmwareVersion']
            self.metadata['InstrumentName'] = data.metadata['InstrumentName']
            self.metadata['InstrumentFamily'] = data.metadata['InstrumentFamily']
            self.metadata['InstrumentRevision'] = data.metadata['InstrumentRevision']
            self.metadata['InstrumentSourceType'] = data.metadata['InstrumentSourceType']
            self.metadata['InstrumentSerialNumber'] = data.metadata['InstrumentSerialNumber']
            self.metadata['Description'] = data.metadata['Description']
            self.metadata['SampleName'] = data.metadata['SampleName']
            self.metadata['MethodName'] = data.metadata['MethodName']
            self.metadata['DenoisingEnabled'] = data.metadata['DenoisingEnabled']
            self.metadata['PeakWidthEstimateValue'] = data.metadata['PeakWidthEstimateValue']
            self.metadata['PeakWidthEstimateType'] = data.metadata['PeakWidthEstimateType']
            self.metadata['HasProfileSpectra'] = data.metadata['HasProfileSpectra']
        except Exception as e:
            custom_warning("There was an issue obtaining some of the metadata from the Bruker .tsf file.", err=e)
        return self.metadata

    # ================================================================================
    # tsf/baf MS2
    # ================================================================================
    # TODO: Verify that MS2 data extraction for Bruker .baf files works as intended
    def bruker_d_ms2_no_mob(self, metadata=None, normalize_img_sizes=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None):
        """
        Data processing for Bruker .tsf/.baf files that contain MS2 data.
        
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

        # get mass windows
        MS1_list, _, MS1_polarity_list, _, _, _, _, mass_list_idxs = self.mass_list
        
        acq_times, all_filters_list = self.check_dim()
        self.metadata['average_start_time'] = np.mean([i[0] for i in acq_times])
        self.metadata['average_end_time'] = np.mean([i[-1] for i in acq_times])

        # for MSMS, extracts info from filters
        filters_info, filter_inverse = self.get_filters_info(all_filters_list)
        # Determines correspondance of MZs to filters
        mzsPerFilter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter \
            = self.get_CountsPerFilter(filters_info)
        # finds the number of scans that use a specific filter
        scans_per_filter = self.get_ScansPerFilter(filters_info, all_filters_list, filter_inverse)
        # Groups filters into groups containing the same mzs/transitions
        consolidated_filter_list, mzs_per_filter_grp, mzs_per_filter_grp_lb, mzs_per_filter_grp_ub, \
            mz_idxs_per_filter_grp, scans_per_filter_grp, peak_counts_per_filter_grp, consolidated_idx_list \
            = self.consolidate_filter_list(filters_info, mzsPerFilter, scans_per_filter, mzsPerFilter_lb, mzsPerFilter_ub, mzIndicesPerFilter)
        num_filter_groups = len(consolidated_filter_list)

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
            
            # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
            TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
            # counts how many times numbers have been inputted each array
            counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

            data = tsf.tsf_data(Name, tsf.dll)
            
            if i == 0:
                # updates the metadata dictionary with instrument information
                self.metadata = get_basic_instrument_metadata(data, self.metadata)

            # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
            pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
            
            frames_info = data.frames
            TIC = frames_info["SummedIntensities"].values

            for j, TimeStamp in tqdm(enumerate(acq_times[i]), disable = True):
                # Update gui variables
                self.progressbar_update_progress(len(acq_times[i]), i, j)

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

                    lb,ub = np.array(mzs_per_filter_grp_lb[grp], dtype = np.float64), np.array(mzs_per_filter_grp_ub[grp], dtype = np.float64)
                
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

                del data

            all_TimeStamps.append(TimeStamps)
            pixels_metas.append(pixels_meta)

        self.rts = acq_times
        pixels, all_TimeStamps_aligned = self.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, mzs_per_filter_grp)

        # Order the pixels in the way the mass list csv/excel file was ordered
        pixels = self.reorder_pixels_d(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)    
        if normalize_img_sizes:
            pixels = self.pixels_list_to_array(pixels, all_TimeStamps_aligned)

        return self.metadata, pixels


    # ================================================================================
    # tdf MS1
    # ================================================================================

    def tdf_d_ms1_mob(self, metadata=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None):
        """
        Data processing from Bruker .tdf files with only MS1 data and ion mobility data.
        
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
        MS1_list, MS1_mob_list, MS1_polarity_list, _, _, _, _, mass_list_idxs = self.mass_list
        lb, mob_lb, _, _, _, _, _ = self.lower_lims
        ub, mob_ub, _, _, _, _, _ = self.upper_lims

        # determine range of mz and mob values to consider
        min_mz, max_mz = np.min(lb), np.max(ub)
        min_mob, max_mob = np.min(mob_lb), np.max(mob_ub)
        
        # monitor progress on gui
        self.progressbar_start_extraction()

        # initiate accumulator
        pixels = []
        rts = []

        # iterate over lines
        for i, file_dir in tqdm(enumerate(self.line_list), total = len(self.line_list), desc='Progress through lines', disable = True): 
            
            # open line data file
            data = OpenTIMS(file_dir)

            if i == 0:
                # updates the metadata dictionary with instrument information
                self.metadata = self.get_basic_instrument_metadata(data, self.metadata)

            # get general data to initialize the line pixels array
            line_rts = data.retention_times
            num_spe = len(line_rts)
            line_pixels = np.zeros((num_spe, len(MS1_list)+1))

            # iterate over spectrum iterator
            for j, frame_dict in tqdm(enumerate(data.query_iter(frames = data.frames['Id'], columns = ('mz', 'inv_ion_mobility', 'intensity'))), \
                                    "progress thru line {linenum} of {total_lines}".format(linenum = i+1, total_lines = len(self.line_list)), total = num_spe, delay = 0.05, disable = (self.testing or self.gui)):
                # Update gui variables
                self.progressbar_update_progress(num_spe, i, j)

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

        self.rts = rts
        pixels_aligned = self.ms1_interp(pixels, mass_list = MS1_list)

        return self.metadata, pixels_aligned 

    def tdf_d_ms2_mob(self, metadata=None, normalize_img_sizes=None, in_jupyter=None, testing=None, gui=None, pixels_per_line=None, tkinter_widgets=None, **kwargs):
        """
        Data processing from Bruker .tdf files that contain MS2 data and ion mobility data.
        
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

        # get mass windows
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

        all_TimeStamps = []
        pixels_metas = []

        # keywords used in extracting data later
        columns_for_data_extraction = ('mz','inv_ion_mobility','intensity')

        # monitor progress on gui
        self.progressbar_start_extraction()

        # holds index of current scan/spectrum
        scan_idx = 0

        for i, file_dir in tqdm(enumerate(self.line_list), desc = 'Progress through lines', total = len(self.line_list), disable = (self.testing or self.gui)):                
            # accumulators for all fitlers,for line before interpolation, interpolation: intensity, scan/acq_time
            TimeStamps = [ np.zeros((scans_per_filter_grp[i][_])) for _ in range(num_filter_groups) ] # spectra for each filter
            # counts how many times numbers have been inputted each array
            counter = np.zeros((scans_per_filter_grp[0].shape[0])).astype(int)-1 # start from -1, +=1 before handeling

            data = OpenTIMS(file_dir)

            if i == 0:
                # updates the metadata dictionary with instrument information
                self.metadata = self.get_basic_instrument_metadata(data, self.metadata)

            # a list of 2d matrix, matrix: scans x (mzs +1)  , 1 -> tic
            pixels_meta = [ np.zeros((scans_per_filter_grp[i][_] , peak_counts_per_filter_grp[_] + 1)) for _ in range(num_filter_groups) ]
            
            frames_info = data.frames
            TIC = data.framesTIC()

            for j, TimeStamp in tqdm(enumerate(acq_times[i]), disable = True):
                # Update gui variables
                self.progressbar_update_progress(len(acq_times[i]), i, j)

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
                    idxs_to_sum = self.vectorized_unsorted_slice_mob(mz,mob,lbs,ubs,mob_lbs,mob_ubs)
                    pixels_meta[grp][counter[grp],1:] = np.sum(np.take(intensity_points, np.array(idxs_to_sum)), axis = 1)

                # keep count of the 1d scan index
                scan_idx += 1

            data.close()
            del data

            all_TimeStamps.append(TimeStamps)
            pixels_metas.append(pixels_meta)
        
        self.rts = acq_times
        pixels, all_TimeStamps_aligned = self.ms2_interp(pixels_metas, all_TimeStamps, acq_times, scans_per_filter_grp, mzs_per_filter_grp)

        # Order the pixels in the way the mass list csv/excel file was ordered
        pixels = self.reorder_pixels_d(pixels, consolidated_filter_list, mz_idxs_per_filter_grp, mass_list_idxs)    
        if normalize_img_sizes:
            pixels = self.pixels_list_to_array(pixels, all_TimeStamps_aligned)

        return self.metadata, pixels