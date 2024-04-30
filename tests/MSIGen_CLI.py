### CLI interface for MSIGen

import argparse, os, re, json, sys
from MSIGen.msigen import get_metadata_and_params, get_image_data
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
    'example_file',
    'mass_list_dir',
    'mass_tolerance_MS1',
    'mass_tolerance_MS1_units',
    'mass_tolerance_prec',
    'mass_tolerance_prec_units',
    'mass_tolerance_frag',
    'mass_tolerance_frag_units',
    'mobility_tolerance',
    'mobility_tolerance_units',
    'img_height',
    'img_width',
    'image_dimensions_units',
    'is_MS2',
    'is_mobility',
    'normalize_img_sizes',
    'output_file_loc',
    'example_file',
    'mass_list_dir',
    'mass_tolerance_MS1',
    'mass_tolerance_MS1_units',
    'mass_tolerance_prec',
    'mass_tolerance_prec_units',
    'mass_tolerance_frag',
    'mass_tolerance_frag_units',
    'mobility_tolerance',
    'mobility_tolerance_units',
    'img_height',
    'img_width',
    'image_dimensions_units',
    'is_MS2',
    'is_mobility',
    'normalize_img_sizes',
    'output_file_loc',
    'scale',
    'aspect',
    'normalize',
    'std_idx',
    'std_precursor',
    'std_mass',
    'std_fragment',
    'std_mobility',
    'std_charge',
    'cmap',
    'titles',
    'images_to_display',
    'threshold',
    'save_images',
    'title_fontsize'
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
        raise ArgumentError("You must provide at least one config file")

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

            # get metadata
            metadata = get_metadata_and_params(argument_dict['example_file'], argument_dict['mass_list_dir'], \
                argument_dict['mass_tolerance_MS1'], argument_dict['mass_tolerance_MS1_units'], \
                argument_dict['mass_tolerance_prec'], argument_dict['mass_tolerance_prec_units'], \
                argument_dict['mass_tolerance_frag'], argument_dict['mass_tolerance_frag_units'], \
                argument_dict['mobility_tolerance'], argument_dict['mobility_tolerance_units'], \
                argument_dict['img_height'], argument_dict['img_width'], argument_dict['image_dimensions_units'], \
                argument_dict['is_MS2'], argument_dict['is_mobility'], argument_dict['normalize_img_sizes'], \
                argument_dict['output_file_loc'], in_jupyter = False, testing = args.testing)
            
            # extract images
            metadata, pixels = get_image_data(metadata, verbose=args.testing, in_jupyter=False, testing=args.testing, gui=False)
            
            if argument_dict['save_images'] is not False:
                # defaults to saving ion images
                if argument_dict['images_to_display'] is None:
                    argument_dict['images_to_display'] = "ion_images"

                # save ion images
                print("saving images to: " + os.path.join(argument_dict['output_file_loc'],'images'))

                if argument_dict['images_to_display'].lower() in ['ion images', 'ion_images']:
                    # get default values for unspecified arguments
                    ion_image_args = ['normalize', 'std_idx', 'std_precursor', 'std_mass', 'std_fragment', \
                        'std_mobility', 'std_charge', 'aspect', 'scale', 'cmap', 'titles', 'threshold', "title_fontsize"]
                    defaults = [None, 1, None, None, None, None, None, None, 0.999, 'viridis', None, None, 10]
                    for i, key in enumerate(ion_image_args):
                        if argument_dict[key] is None:
                            argument_dict[key] = defaults[i]
                    
                    # get and save images
                    pixels_normed = vis.get_pixels_to_display(pixels, metadata, argument_dict['normalize'], argument_dict['std_idx'], \
                        argument_dict['std_precursor'], argument_dict['std_mass'], argument_dict['std_fragment'], \
                        argument_dict['std_mobility'], argument_dict['std_charge'])
                    
                    vis.save_images(pixels_normed, metadata, argument_dict['aspect'], argument_dict['scale'], \
                        argument_dict['output_file_loc'], argument_dict['cmap'], argument_dict['titles'], \
                        argument_dict['threshold'], title_fontsize = argument_dict['title_fontsize']) 

                # save fractional abundance images
                elif argument_dict['images_to_display'].lower() in ['fract_abund', "fractional_abundance_images", \
                    "fractional abundance images", 'fract', 'fractional images', 'fractional_images', 'fract abund', \
                    "fract_images", 'fract_image', "fract images", 'fract image', 'fraction', 'fractional']:
                    
                    #get defaults for unspecified args
                    ion_image_args = ['normalize', 'std_idx', 'aspect', 'scale', 'cmap', 'titles', 'title_fontsize']
                    defaults = [None, [1,2], None, 0.999, 'viridis', None, 10]
                    for i, key in enumerate(fract_image_args):
                        if argument_dict[key] is None:
                            argument_dict[key] = defaults[i]

                    # get and save images
                    fractional_abundance_images(pixels, metadata, idxs = argument_dict['std_idxs'], normalize = argument_dict['normalize'], \
                        titles = argument_dict['titles'], aspect = argument_dict['aspect'], save_imgs = True, \
                        MSI_data_output = argument_dict['output_file_loc'], cmap = argument_dict['cmap'], title_fontsize = argument_dict['title_fontsize'])

                # save ratio images
                elif argument_dict['images_to_display'].lower() in ['ratio', "ratio_images", "ratio images", "ratio_image",  "ratio_image", \
                    'ratio_img', 'ratio img']:

                    # get defaults for unspecified args
                    ion_image_args = ['normalize', 'std_idx', 'aspect', 'scale', 'cmap', 'titles', 'handle_infinity', \
                                        'log_scale', 'threshold', 'title_fontsize']
                    defaults = [None, [1,2], None, 0.999, 'viridis', None, 'maximum', False, None, 10]
                    for i, key in enumerate(fract_image_args):
                        if argument_dict[key] is None:
                            argument_dict[key] = defaults[i]

                    #get and save imgs
                    ratio_images(pixels, metadata, idxs = argument_dict['std_idxs'], normalize = argument_dict['normalize'], \
                        handle_infinity = argument_dict['handle_infinity'], titles = argument_dict['titles'], \
                        aspect = argument_dict['aspect'], save_imgs = True, MSI_data_output = argument_dict['output_file_loc'], \
                        cmap = argument_dict['cmap'], log_scale = argument_dict['log_scale'], threshold = argument_dict['threshold'],\
                        title_fontsize=argument_dict['title_fontsize'])
        
        except Exception as error:
            print(f"An exception occurred while processing:\n{file}\n", type(error).__name__, "–", error) 