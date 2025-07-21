### CLI interface for MSIGen

import argparse, os, re, json, sys
from MSIGen import msigen
from MSIGen import visualization as vis

# allows for logging outputs to file
class Logger(object):
    """Allows for logging stdout to file."""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.split(__file__)[0]+'\\logfile.txt', "a")
        print("Log file is being written at " + os.path.split(__file__)[0]+'\\logfile.txt')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

def fix_improper_backslashes(config_file_path):
    """Replaces single backslashes in the argument .json files with double backslashes"""
    settings_tmp_file = os.path.join(os.path.split(config_file_path)[0], "~temp-"+os.path.split(config_file_path)[1])
    with open(config_file_path, "r") as in_file:
        with open(settings_tmp_file,"w") as outfile:
            # print(in_file.read())
            text = in_file.read()
            text = re.sub(r"(?!\\)\\(?!\\)(?!u03bcs)",r'\\\\',text)
            outfile.write(text)
    os.remove(config_file_path)
    os.rename(settings_tmp_file, config_file_path)

# list of possible vars:
possible_vars = [
    'example_file', # If a single file is provided, all files wiith the same name apart from the line number will be processed. If multiple files are provided, all provided files will be processed.
    'mass_list_dir', # Directory containing mass list in .csv or excel format
    'mass_tolerance_MS1', # Mass tolerance for MS1 in ppm or m/z units
    'mass_tolerance_MS1_units', # Units for mass tolerance for MS1, either 'ppm' or 'm/z'
    'mass_tolerance_prec', # Mass tolerance for precursor ions in ppm or m/z units
    'mass_tolerance_prec_units', # Units for mass tolerance for precursor ions, either 'ppm' or 'm/z'
    'mass_tolerance_frag', # Mass tolerance for fragment ions in ppm or m/z units
    'mass_tolerance_frag_units', # Units for mass tolerance for fragment ions, either 'ppm' or 'm/z'
    'mobility_tolerance', # Mobility tolerance in μs or inverse ion mobility (1/k0)
    'mobility_tolerance_units', # Units for mobility tolerance, either '1/k0' or 'μs'
    'img_height', # Height of the imaged area
    'img_width', # Width of the imaged area
    'image_dimensions_units', # Units for image dimensions, usually 'mm'
    'is_MS2', # Whether the files contains any MS2 data or not, True or False
    'is_mobility', # Whether the files contains any mobility data or not, True or False
    'normalize_img_sizes', # if using ms2 data, True will make all images the same size, False will use the original image sizes. Numpy save format will be npz instead of npy if False.
    'pixels_per_line', # How to determine the number of pixels per line in the image. Can be mean, max, min, or an integer value.
    'output_file_loc', # Location to save output files
    'scale', # The quantile of pixel intensities to scale the image intensities to, between 0 and 1. Lower numbers will result decrease the intensity of the brightest pixels, leading to a brighter image. Default is 0.999.
    'aspect', # Aspect ratio for the images. Automatically determined using img_height and img_width if None, otherwise can be a float value.
    'normalize', # How to normalize the MS images. Can be 'None', 'TIC', or 'intl_std'.
    'std_idx', # The index in the mass list of the internal standard to use for normalization. If None, this will be determined based on std_precursor, std_mass, std_fragment, std_mobility, and std_charge.
    'std_precursor', # The mass of the internal standard precursor ion. Unused if std_idx is not None.
    'std_mass', # The mass of the internal standard fragment ion. Unused if std_idx is not None.
    'std_fragment', # The mass of the internal standard fragment ion. Unused if std_idx is not None.
    'std_mobility', # The mobility of the internal standard ion. Unused if std_idx is not None.
    'std_charge', # The charge state of the internal standard ion. Unused if std_idx is not None.
    'cmap', # Colormap to use for displaying the images. Default is 'viridis'. Can be any matplotlib colormap.
    'how_many_images_to_display', # How many images to display. can be 'all', an integer, or a list of integers. If 'all', all images will be displayed. If an integer, that many images will be displayed. If a list, the images at the specified indices will be displayed.
    'titles', # Titles for the images. If None, the titles will be automatically generated. If a list, the titles will be used for the images.
    'type_of_images_to_display', # The type of images to display. Can be 'ion_images', 'fractional_abundance_images', 'ratio_images', or None. If None, ion images will be displayed.
    'images_to_display', # Deprecated alias of 'type_of_images_to_display', use 'type_of_images_to_display' instead. Will be removed in future versions.
    'fract_img_idxs', # List of indices corresponding to the images to use for the fractional abundance images. If None, the first two images will be used. Ignored if type_of_images_to_display is not 'fractional_abundance_images'.
    'ratio_img_idxs', # List of indices corresponding to the images to use for the ratio images. If None, the first two images will be used. Ignored if type_of_images_to_display is not 'ratio_images'.
    'log_scale', # Whether to use a logarithmic scale for the ratio images. Default is False. If True, the ratio images will be displayed on a logarithmic scale.
    'handle_infinity', # How to handle infinity values in the ratio images. Can be 'maximum', 'zero', or 'infinity'. Infinity values will be replaced with the either the maximum value in the image or the specified value.
    'threshold', # Reduces the intensity of any pixels above this threshold to this value when displaying the images. If None, no threshold will be applied.
    'save_images', # Whether to save the images or not. If True, the images will be saved to the output file location.
    'title_fontsize', # Font size for the titles. If None, the default font size is 10.
    'image_savetype', # The type of image to save. Options are 'figure', 'image', or 'array'. 'figure' will save the image as a figure with a colorbar and title. 'image' will save the image as an image without a colorbar or title. 'array' will save the image as an array in csv format.
    'axis_tick_marks', # Whether to display tick marks on the edges of the image. Default is False.
    'interpolation', # Interpolation method for displaying the images. Default is 'nearest'. Can be any matplotlib interpolation method.
    'fig_height', # Height of the figure in inches for publication-style figures. Default is 6.
    'fig_width' # Width of the figure in inches for publication-style figures. Default is 6.
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple file paths.")
    parser.add_argument("filepaths", nargs="+", help="Paths of files to process")
    parser.add_argument("-t", "--testing", action="store_true", help="Enable testing mode")

    args = parser.parse_args()

    # print outputs to file for debugging
    if args.testing:
        logger = Logger()
        sys.stdout = logger

    # account for incorrect filepath arguments
    if len(args.filepaths) == 0:
        raise ValueError("You must provide at least one config file")

    for file in args.filepaths:
        print(file)
        try:
            if not os.path.exists(file):
                print(f"File does not exist")
                continue
            if not file.endswith('.json'):
                print(f"file must be in .json format")
                continue
            
            # initialize argument dict
            argument_dict = {}
            for i in possible_vars:
                argument_dict[i] = None

            # Fix '\' errors within files
            fix_improper_backslashes(file)
            with open(file) as f:
                metadata = json.load(f)
            
            # update arguments with those contained in metadata
            argument_dict = argument_dict | metadata

            # Initialize the MSIGen object
            MSIGen_generator = msigen(example_file=argument_dict['example_file'], mass_list_dir=argument_dict['mass_list_dir'], \
                tol_MS1=argument_dict['mass_tolerance_MS1'], tol_MS1_u=argument_dict['mass_tolerance_MS1_units'], \
                tol_prec=argument_dict['mass_tolerance_prec'], tol_prec_u=argument_dict['mass_tolerance_prec_units'], \
                tol_frag=argument_dict['mass_tolerance_frag'], tol_frag_u=argument_dict['mass_tolerance_frag_units'], \
                tol_mob=argument_dict['mobility_tolerance'], tol_mob_u=argument_dict['mobility_tolerance_units'], \
                h=argument_dict['img_height'], w=argument_dict['img_width'], hw_units=argument_dict['image_dimensions_units'], \
                is_MS2=argument_dict['is_MS2'], is_mobility=argument_dict['is_mobility'], \
                normalize_img_sizes=argument_dict['normalize_img_sizes'], pixels_per_line=argument_dict['pixels_per_line'], \
                output_file_loc=argument_dict['output_file_loc'], in_jupyter = False, testing = args.testing)
            
            # extract images
            metadata, pixels = MSIGen_generator.get_image_data(verbose=args.testing, testing=args.testing, gui=False)
            
            if argument_dict['save_images'] is not False:
                # defaults to saving ion images
                if argument_dict['type_of_images_to_display'] is None:
                    if argument_dict['images_to_display'] is not None:
                        print("WARNING: 'images_to_display' is deprecated, use 'type_of_images_to_display' instead.")
                        argument_dict['type_of_images_to_display'] = argument_dict['images_to_display']
                    argument_dict['type_of_images_to_display'] = "ion_images"

                # save ion images
                print("saving images to: " + os.path.join(argument_dict['output_file_loc'],'images'))

                if argument_dict['type_of_images_to_display'].lower() in ['ion images', 'ion_images']:
                    # get default values for unspecified arguments
                    ion_image_args = ['normalize', 'std_idx', 'std_precursor', 'std_mass', 'std_fragment', \
                        'std_mobility', 'std_charge', 'aspect', 'scale', 'how_many_images_to_display', 'cmap', 'titles', 'threshold', \
                        'title_fontsize', 'image_savetype', 'axis_tick_marks', 'interpolation', 'fig_height', 'fig_width']
                    defaults = [None, 1, None, None, None, None, None, None, 0.999, 'all', 'viridis', None, None, 10,'figure',False, 'none', 6, 6]
                    for i, key in enumerate(ion_image_args):
                        if argument_dict.get(key) is None:
                            argument_dict[key] = defaults[i]
                    
                    # get and save images
                    pixels_normed = vis.get_pixels_to_display(pixels, metadata, argument_dict['normalize'], argument_dict['std_idx'], \
                        argument_dict['std_precursor'], argument_dict['std_mass'], argument_dict['std_fragment'], \
                        argument_dict['std_mobility'], argument_dict['std_charge'])
                    
                    vis.display_images(pixels_normed, metadata, aspect=argument_dict['aspect'], scale=argument_dict['scale'], \
                        how_many_images_to_display=argument_dict['how_many_images_to_display'], save_imgs=True, \
                        MSI_data_output=argument_dict['output_file_loc'], cmap=argument_dict['cmap'], titles=argument_dict['titles'], \
                        threshold=argument_dict['threshold'], title_fontsize=argument_dict['title_fontsize'], \
                        image_savetype=argument_dict['image_savetype'], axis_tick_marks=argument_dict['axis_tick_marks'],
                        interpolation=argument_dict['interpolation'], h=argument_dict['fig_height'], w=argument_dict['fig_width'])

                # save fractional abundance images
                elif argument_dict['type_of_images_to_display'].lower() in ['fract_abund', "fractional_abundance_images", \
                    "fractional abundance images", 'fract', 'fractional images', 'fractional_images', 'fract abund', \
                    "fract_images", 'fract_image', "fract images", 'fract image', 'fraction', 'fractional']:
                    
                    #get defaults for unspecified args
                    ion_image_args = ['fract_img_idxs', 'normalize', 'titles', 'aspect', 'cmap', 'title_fontsize', 'image_savetype', 'scale', 'threshold', 'axis_tick_marks', 'interpolation', 'fig_height', 'fig_width']
                    defaults = [[1,2], None, None, None, 'viridis', 10, 'figure', 1, None, False, 'none', 6, 6]
                    for i, key in enumerate(ion_image_args):
                        if argument_dict[key] is None:
                            argument_dict[key] = defaults[i]

                    # get and save images
                    vis.fractional_abundance_images(pixels, metadata, idxs = argument_dict['fract_img_idxs'], normalize = argument_dict['normalize'], \
                        titles = argument_dict['titles'], aspect = argument_dict['aspect'], save_imgs = True, \
                        MSI_data_output = argument_dict['output_file_loc'], cmap = argument_dict['cmap'], \
                        title_fontsize = argument_dict['title_fontsize'], image_savetype = argument_dict['image_savetype'], \
                        scale = argument_dict['scale'], threshold = argument_dict['threshold'], axis_tick_marks=argument_dict['axis_tick_marks'], \
                        interpolation = argument_dict['interpolation'], h=argument_dict['fig_height'], w=argument_dict['fig_width'])

                # save ratio images
                elif argument_dict['type_of_images_to_display'].lower() in ['ratio', "ratio_images", "ratio images", "ratio_image",  "ratio_image", \
                    'ratio_img', 'ratio img']:

                    # get defaults for unspecified args
                    ion_image_args = ['ratio_img_idxs', 'normalize', 'handle_infinity', 'titles', 'aspect', 'scale', 'cmap', \
                                      'log_scale', 'threshold', 'title_fontsize', 'image_savetype', 'axis_tick_marks', \
                                      'interpolation', 'fig_height', 'fig_width']
                    defaults = [[1,2], None, 'maximum', None, None, 0.999, 'viridis', False, None, 10, 'figure', False, 'none', 6, 6]
                    for i, key in enumerate(ion_image_args):
                        if argument_dict[key] is None:
                            argument_dict[key] = defaults[i]

                    #get and save imgs
                    vis.ratio_images(pixels, metadata, idxs = argument_dict['ratio_img_idxs'], normalize = argument_dict['normalize'], \
                        handle_infinity = argument_dict['handle_infinity'], titles = argument_dict['titles'], \
                        aspect = argument_dict['aspect'], scale = argument_dict['scale'], save_imgs = True, \
                        MSI_data_output = argument_dict['output_file_loc'], cmap = argument_dict['cmap'], \
                        log_scale = argument_dict['log_scale'], threshold = argument_dict['threshold'],\
                        title_fontsize=argument_dict['title_fontsize'], image_savetype=argument_dict['image_savetype'], \
                        axis_tick_marks=argument_dict['axis_tick_marks'], interpolation=argument_dict['interpolation'], \
                        h=argument_dict['fig_height'], w=argument_dict['fig_width'])
        
        except Exception as error:
            print(f"An exception occurred while processing:\n{file}\n", type(error).__name__, "-", error) 