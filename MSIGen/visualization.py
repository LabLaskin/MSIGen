"""
Functions used for visualizing images from data processed by MSIGen.
This includes functions for saving and displaying normalized or raw ion images, fractional abundance images, and ratio images.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize
import os
from PIL import Image


# ===========================================================================================
# Raw or normalized image visualization
# ===========================================================================================

def get_normalize_value(normalize, possible_entries = ['None', 'TIC', 'intl_std', 'base_peak']):
    """Parses the value of the normalize variable. Allows for error handling and for some leeway in mistyping the keywords.
    
    Args:
        normalize (str or None): 
            The normalization method. Options are 'None', 'TIC', 'intl_std', or 'base_peak'.
        possible_entries (list): 
            The allowed entries for the normalize variable. Only used to restrict the options for fractional abundance or ratio images.
    """

    if normalize in [None, False]:
        normalize = 'None'

    elif type(normalize) == str:
        normalize_vals_dict = {
            'None': ['none', 'no', 'false'],
            'TIC': ['tic', 'total ion current', 'total_ion_current'],
            'intl_std': ['intl', 'internal', 'internal standard', "internal_standard", 'intl std', 'intl_std', 'standard', 'std'],
            'base_peak': ['base', 'base_peak', 'base peak', 'tallest_peak', 'tallest peak'],
        }                

        # Check if the given value is in the dict
        for key in possible_entries:
            if normalize.lower() in normalize_vals_dict[key]:
                normalize = key
                break
        # raise error if not in dict
        if normalize not in possible_entries:
            raise ValueError(f"Value for 'normalize' should be in {possible_entries}")
    
    else:
        raise ValueError(f"Value for 'normalize' should be one of the following:\n{possible_entries}")
    return normalize

def match_to_mass_list(mass_list, idx = None, precursor = None, mass = None, fragment = None, mobility = None, charge = None):
    """
    Matches the given mass, mobility, or charge values to the mass list. Returns the index of the match in the mass list.
    Fails if there is more than one match or if there is no match.
    If the index is given, the function will return that index.
    If the index is not given, the function will search for an entry in the mass list that uniquely matches the given mass, mobility, and charge values.
    """
    # use given idx if possible
    if idx != None:
        return idx

    # determine what mass arrays should look like based on given values
    elif any(np.array((precursor, mass, fragment, mobility, charge))!=None):
        comparison_arrays = []
        
        # if the value has both precursor and mass or precursor and fragment, check only for ms2 values
        if all(np.array((precursor, mass))!=None):
            comparison_arrays.append(np.array([precursor, mass, mobility, charge]))
        elif all(np.array((precursor, fragment))!=None):
            comparison_arrays.append(np.array([precursor, mass, mobility, charge]))
        
        # otherwise check for ms2 or ms1 scans
        else:
            if mass == None and fragment!=None:
                mass = fragment
            elif mass == None and precursor!=None:
                mass = precursor

            comparison_arrays.append(np.array([precursor, mass, mobility, charge]))
            comparison_arrays.append(np.array([mass, mobility, charge]))

        for i, mass_from_list in mass_list:
            matches = np.zeros(len(mass_list))

            for comparison_mass in comparison_arrays:
                if len(mass_from_list) == len(comparison_mass):
                    matches[i] = np.sum(np.array(mass_from_list)==comparison_mass)
                    break
                else:
                    matches[i] = 0 
        
        num_matches = np.max(matches)
        best_match = np.argwhere(matches == np.max(matches))
        
        assert num_matches == 0, 'There were no matches to the given mass, mobility, or charge values'
        assert best_match.shape != 1, 'There were multiple entries on your mass list that matched the given mass, mobility, or charge values'
        
        idx = best_match.item()
        return idx

    else:
        assert any(np.array((idx, precursor, mass, fragment, mobility, charge))!=None), "At lease one of the following kwargs must be defined: \
        idx, precursor, mass, fragment, mobility, charge"

def normalize_pixels(pixels, std_idx, handle_infinity = 'zero'):
    """
    Normalizes the pixels to the standard image.
    If the images are not all the same size, the standard image will be resized to match the size of the other images.

    Args:
        pixels (list or array): The images to be normalized.
        std_idx (int): The index of the standard image. 0 indicates the TIC image.
    
    Returns:
        pixels_normed (list or array): The normalized images.
    """
    # if the pixels are in a list, normalize them individually because their shapes are likely not all the same 
    if type(pixels) == list:
        pixels_normed=[]
        std_img = pixels[std_idx]
        
        for i in pixels:
            # ensure the images are the same size
            if i.shape != std_img.shape:
                std_img_tmp = resize(std_img, i.shape, order=0)
            else: 
                std_img_tmp = std_img

        if handle_infinity == 'zero':
            out_arr = np.zeros_like(i)
        elif handle_infinity == 'maximum':
            out_arr = np.full_like(i, np.max(i))
        elif handle_infinity == 'infinity':
            out_arr = np.full_like(i, np.inf)
        else:
            raise ValueError("handle_infinity must be 'zero', 'maximum', or 'infinity'")
        pixels_normed.append(np.divide(i, std_img_tmp, out=out_arr, where=std_img_tmp!=0))
    
    # If the pixels are in an array, normalize them all together
    elif type(pixels) == type(np.array(0)):
        if handle_infinity == 'zero':
            out_arr = np.zeros_like(pixels)
        elif handle_infinity == 'maximum':
            # an array that has the maximum value for each image as the default value fot that image
            out_arr = np.array([np.full_like(pixels[i], np.max(pixels[i])) for i in range(pixels.shape[0])])
        elif handle_infinity == 'infinity':
            out_arr = np.full_like(pixels, np.inf)
        pixels_normed = np.divide(pixels, pixels[std_idx], out=out_arr, where=pixels[std_idx]!=0)
    
    return pixels_normed

def base_peak_normalize_pixels(pixels):
    """Normalizes each image to the highest intensity in pixels in that image."""
    for i, img in enumerate(pixels):
        if img.max():
            pixels[i]=img/img.max()
    return pixels

def despike_images(pixels, threshold = 1.5, num_pixels_on_each_side = 2, axis = 'x'):
    """
    Despikes the images by comparing the pixel value to the mean of the surrounding pixels.
    If the pixel value is greater than the mean of the surrounding pixels by a certain threshold, it is replaced with the mean.
    Despiking is done only along the x-axis by default.
    """
    n = num_pixels_on_each_side
    output_arr = []
    for img in pixels:
        if axis == 'x':
            comparison_arr = np.zeros((img.shape[0], img.shape[1] - 2*n))
            for i in range(img.shape[1]):
                if i < n or i >= img.shape[1] - n:
                    continue
                column = img[:, i-n:i+n+1]
                column = np.delete(column, n, axis=1)
                column = np.array([np.mean(i[i!=0]) if i[i!=0].shape[0] else 0 for i in column])
                column[column == 0] = img[column == 0, i]
                comparison_arr[:, i-n] = column
        elif axis == 'y':
            raise NotImplementedError("Despiking along the y-axis is not yet implemented")
        
        output_arr.append(np.where(img[:, n:-n] > comparison_arr*threshold, comparison_arr, img[:, n:-n]))
    return output_arr

def get_and_display_images(pixels, metadata=None, normalize = None, std_idx = None, std_precursor = None, std_mass = None, \
                        std_fragment = None, std_mobility = None, std_charge = None, aspect = None, scale = .999, \
                        how_many_images_to_display = 'all', save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                        titles = None, threshold = None, title_fontsize = 10, image_savetype = "figure", \
                        axis_tick_marks = False, interpolation='none', h = 6, w = 6, handle_infinity = 'zero'):
    """
    Displays the images in the pixels array. The images are normalized to the standard image or to the TIC image.
    
    Args:
        pixels (list or array): 
            The images to be displayed.
        metadata (dict): 
            The metadata for the images. This should include the mass list and the image dimensions.
        normalize (str): 
            The normalization method. Options are 'None', 'TIC', or 'intl_std'.
        std_idx (int): 
            The index of the standard image. Ignored unless normalize is 'intl_std'.
            If none, the std_idx will be determined based on std_precursor, std_mass, std_fragment, std_mobility, and std_charge.
            0 indicates the TIC image.
        std_precursor (float): 
            The precursor mass of the standard. Ignored if std_idx is given.
        std_mass (float): 
            The mass of the standard. Ignored if std_idx is given.
        std_fragment (float): 
            The fragment mass of the standard. Ignored if std_idx is given.
        std_mobility (float): 
            The mobility of the standard. Ignored if std_idx is given.
        std_charge (int):
            The charge of the standard. Ignored if std_idx is given.
        aspect (float):
            The aspect ratio of each pixel for display. If None, the aspect ratio will be calculated based on the image dimensions.
        scale (float):
            The quantile to lower intensity values to. Default is .999.
            Any pixel with an intensity greater than the pixel with this quantile will be decreased to this intensity. 
            This is done to prevent saturation of the color scale.
            Ignored if threshold is given.
        how_many_images_to_display (int, list, or str):
            The number of images to display if this is an int.
            If this is a list, the images at the indices in the list will be displayed.
            If this is a string, it must be 'all', and all images will be displayed.
        save_imgs (bool):
            If True, the images will be saved to the MSI_data_output directory.
        MSI_data_output (str):
            The directory to save the images to. If None, the images will be saved to the current working directory.
        cmap (str):
            The colormap to use for the images. Default is 'viridis'.
        titles (list):
            The titles for the images. If None, the titles will be determined based on the mass list.
        threshold (float):
            The threshold for the images. Any pixel with an intensity greater than the threshold will be decreased to the threshold.
            If None, the threshold will be determined based on the scale.
        title_fontsize (int):
            The font size of the titles. Default is 10.
        image_savetype (str):
            The type of image to save. Options are 'figure', 'image', or 'array'.
            'figure' will save the image as a figure with a colorbar and title.
            'image' will save the image as an image without a colorbar or title.
            'array' will save the image as an array in csv format.
        axis_tick_marks (bool):
            If True, the axis tick marks will be shown. Default is False.
        interpolation (str):
            The interpolation method to use for displaying the images. Default is 'none'.
            Using 'nearest' or 'none' will make the images look pixelated, while 'bilinear' will make them look smoother/blurrier.
            See https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html for more options.
        """
    pixels_normed = get_pixels_to_display(pixels, metadata, normalize, std_idx, std_precursor, std_mass, std_fragment, std_mobility, std_charge, handle_infinity)
    display_images(pixels_normed, metadata, aspect, scale, how_many_images_to_display, save_imgs, MSI_data_output, cmap, titles, threshold, \
                   title_fontsize, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks, interpolation=interpolation, h=h, w=w)

def get_pixels_to_display(pixels, metadata=None, normalize = None, std_idx = None, std_precursor = None, std_mass = None, std_fragment = None, std_mobility = None, std_charge = None, handle_infinity = 'zero'):
    """
    Normalizes pixels to TIC or to an internal standard.
    The if images are of varying size, the standard image is reshaped to the size of the image to be normalized.
    """

    normalize = get_normalize_value(normalize)

    if metadata == None:
        mass_list = [0.0 for i in range(len(pixels))]
        raise Warning("No metadata provided. Mass list will be set to all zeros and image titles will not be specific.")
    else:
        mass_list = metadata["final_mass_list"]

    if normalize == 'intl_std':
        # find the index of the standard
        std_idx = match_to_mass_list(mass_list, std_idx, std_precursor, std_mass, std_fragment, std_mobility, std_charge)
        pixels_normed = normalize_pixels(pixels, std_idx, handle_infinity=handle_infinity)

    elif normalize == 'TIC':
        pixels_normed = normalize_pixels(pixels, 0, handle_infinity=handle_infinity)
    
    elif normalize == 'base_peak':
        pixels_normed = base_peak_normalize_pixels(pixels)

    else:
        pixels_normed = pixels

    return pixels_normed

def display_images(pixels_normed, metadata=None, aspect = None, scale = .999, how_many_images_to_display = 'all', \
                    save_imgs = False, MSI_data_output = None, cmap = 'viridis', titles = None, threshold = None, \
                    title_fontsize = 10, image_savetype = "figure", axis_tick_marks = False, interpolation='none', \
                    h = 6, w = 6):
    """
    Displays the images in the pixels array. Normalization must be performed prior to calling this.
    """

    # parse args
    if how_many_images_to_display == 'all':
        how_many_images_to_display = len(pixels_normed)
    if type(how_many_images_to_display) in [str, int, float]:
        try:
            how_many_images_to_display = list(range(int(how_many_images_to_display)))
        except:
            raise TypeError("how_many_images_to_display must be 'all', an integer, or a list of integers")
    if type(how_many_images_to_display) in [list, tuple]:
        try:
            how_many_images_to_display = [int(i) for i in how_many_images_to_display]
        except:
            raise TypeError("how_many_images_to_display must be 'all', an integer, or a list of integers")

    # Get the titles for all figures:
    if metadata == None:
        default_titles = ["Image "+str(i) for i in how_many_images_to_display]
    else:
        mass_list = metadata["final_mass_list"]
        default_titles = determine_titles(mass_list, idxs = how_many_images_to_display)

    # make sure save directory exists
    if MSI_data_output == None:
        MSI_data_output = os.getcwd()
    img_output_folder = os.path.join(MSI_data_output,'ion_images')
    if save_imgs:
        if not os.path.exists(img_output_folder):
            os.makedirs(img_output_folder)

    # plot each image
    if metadata == None:
        img_height, img_width = 1.0, 1.0
    else:
        img_height, img_width = metadata['image_dimensions']
    # use manually given aspect ratio
    a = aspect

    if threshold:
        thre = threshold

    for i, img_idx in enumerate(how_many_images_to_display):
        # stop early if desired
        img = pixels_normed[img_idx]
        
        if not threshold:
            thre = np.quantile(img, scale)
        if thre == 0: thre = 1

        if titles == None:
            title = default_titles[i]
        else:
            title = titles[i]
        default_title = default_titles[i]
        
        # recalculate aspect ratio for each image
        if aspect == None:
            a = (img_height/img.shape[0])/(img_width/img.shape[1])

        plot_image(img=img, img_output_folder=img_output_folder, title=title, default_title=default_title, title_fontsize=title_fontsize, \
                cmap=cmap, aspect=a, save_imgs=save_imgs, thre=thre, log_scale = False, image_savetype=image_savetype, \
                axis_tick_marks=axis_tick_marks, interpolation=interpolation, h=h, w=w)


def determine_titles(mass_list, idxs = None, fract_abund = False, ratio_img=False):
    """
    Function for determining the default titles for the images.

    Args:
        mass_list (list): The mass list for the images.
        idxs (list): The indices of the images to generate titles for. If None, titles will be generated for all images.
        fract_abund (bool): If True, the titles will be for fractional abundance images.
        ratio_img (bool): If True, the titles will be for ratio images.
    """
    titles = []
    polarity_dict = { 1.0:'+',
                    0.0:'',
                    -1.0:'-'}

    if idxs == None:
        idxs = range(len(mass_list))

    for i in idxs:
        entry = mass_list[i]
        if len(entry) == 1:
            titles.append("TIC")
        else:
            if len(entry)==3:
                title_mass = 'm/z:' + str(round(entry[0], 4))
            elif len(entry)==4:
                title_mass =  str(round(entry[0], 4))+' -> '+str(round(entry[1], 4))
            if entry[-2]:
                title_mob = '\nMobility:' + str(round(entry[1], 4))
            else: 
                title_mob = ''
            title_polarity = polarity_dict[entry[-1]]
            if fract_abund:
                titles.append('Fractional abundance of\n'+title_polarity+title_mass+title_mob)
            elif ratio_img:
                titles.append('Ratio image of\n'+title_polarity+title_mass+title_mob)
            else:
                titles.append(title_polarity+title_mass+title_mob)
    return titles


# ===========================================================================================
# fractional abuncance images
# ===========================================================================================

def fractional_abundance_images(pixels, metadata=None, idxs = [1,2], normalize = None, titles = None, \
                        aspect = None, save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                        title_fontsize = 10, image_savetype = 'figure', scale = 1.0, threshold = None, \
                        axis_tick_marks = False, interpolation = 'none', h = 6, w = 6):
    """
    Generates fractional abundance images from the given pixels, metadata, and indices.
    The images are divided by the sum of the images to get the fractional abundance.
    
    Args:
        pixels (list or array): 
            The images to be displayed.
        metadata (dict): 
            The metadata for the images. This should include the mass list and the image dimensions.
        idxs (list):
            The indices of the images to be used.
        normalize (str): 
            The normalization method. Options are 'None', or 'base_peak'.
            'base_peak' will normalize the images to the base peak intensity before division.
        titles (list):
            The titles for the images. If None, the titles will be determined based on the mass list.
        aspect (float):
            The aspect ratio of each pixel for display. If None, the aspect ratio will be calculated based on the image dimensions.
        save_imgs (bool):
            If True, the images will be saved to the MSI_data_output directory.
        MSI_data_output (str):
            The directory to save the images to. If None, the images will be saved to the current working directory.
        cmap (str):
            The colormap to use for the images. Default is 'viridis'.
        title_fontsize (int):
            The font size of the titles. Default is 10.
        image_savetype (str):
            The type of image to save. Options are 'figure', 'image', or 'array'.
            'figure' will save the image as a figure with a colorbar and title.
            'image' will save the image as an image without a colorbar or title.
            'array' will save the image as an array in csv format.
        scale (float):
            The quantile to lower intensity values to. Default is .999.
            Any pixel with an intensity greater than the pixel with this quantile will be decreased to this intensity. 
            This is done to prevent saturation of the color scale.
            Ignored if threshold is given.
        threshold (float):
            The threshold for the images. Any pixel with an intensity greater than the threshold will be decreased to the threshold.
            If None, the threshold will be determined based on the scale.
        axis_tick_marks (bool):
            If True, the axis tick marks will be shown. Default is False.
        interpolation (str):
            The interpolation method to use for displaying the images. Default is 'none'.
            Using 'nearest' or 'none' will make the images look pixelated, while 'bilinear' will make them look smoother/blurrier.
            See https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html for more options.
    """
    
    fract_imgs = get_fractional_abundance_imgs(pixels, metadata, idxs, normalize)
    display_fractional_images(fract_imgs, metadata, titles, aspect, save_imgs, MSI_data_output, cmap, \
                              title_fontsize, idxs, image_savetype=image_savetype, scale=scale, \
                              threshold=threshold, axis_tick_marks=axis_tick_marks, \
                              interpolation=interpolation, h=h, w=w)

def get_fractional_abundance_imgs(pixels, metadata=None, idxs = [1,2], normalize = None):
    """
    Normalizes pixels before getting fractional abundance. 
    The images are divided by the sum of the images to get the fractional abundance.
    If the images are of varying size, all images are resized to the image corresponding to the first index given.

    Returns:
        fract_imgs (list): The fractional abundance images.
    """

    normalize = get_normalize_value(normalize, possible_entries=['None', 'base_peak'])

    imgs = [pixels[i] for i in idxs]

    # Ensure images are all the same size
    shapes = [img.shape for img in imgs]
    idxs_to_reshape = np.any(~np.equal(shapes[0],shapes), axis = 1)
    for idx, i in enumerate(idxs_to_reshape):
        if i: imgs[idx] = resize(imgs[idx], shapes[0], order=0)
    
    if normalize == "base_peak":
        imgs = base_peak_normalize_pixels(imgs)

    img_sum = np.sum(imgs, axis = 0)

    fract_imgs = []
    for i in imgs:
        fract_imgs.append(np.divide(i, img_sum, out=np.zeros_like(i), where=img_sum!=0))

    return fract_imgs

def display_fractional_images(fract_imgs, metadata=None, titles = None, aspect = None,\
                            save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                            title_fontsize = 10, idxs = [1,2], image_savetype='figure', \
                            scale = 1.0, threshold = None, axis_tick_marks = False, \
                            interpolation = 'none', h = 6, w = 6):

    """
    Displays the fractional abundance images in the fract_imgs array.
    """

    if metadata == None:
        default_titles = ["Image "+str(i) + "/Sum of Images" for i in range(len(fract_imgs))]
    else:
        mass_list = metadata["final_mass_list"]
        default_titles = determine_titles(mass_list, idxs = idxs, fract_abund=True)

    # make sure save directory exists
    if MSI_data_output == None:
        MSI_data_output = os.getcwd()
    img_output_folder = os.path.join(MSI_data_output,'fract_images')
    if save_imgs:
        if not os.path.exists(img_output_folder):
            os.makedirs(img_output_folder)

    if threshold:
        thre = threshold

    # plot each image
    if metadata == None:
        img_height, img_width = 1.0, 1.0
    else:
        img_height, img_width = metadata['image_dimensions']
    # use manually given aspect ratio
    a = aspect

    for i in range(len(fract_imgs)):
        
        img = fract_imgs[i]

        if not threshold:
            thre = np.quantile(img, scale)
        if thre == 0: thre = 1

        if titles == None:
            title = default_titles[i]
        else:
            title = titles[i]
        default_title = default_titles[i]
        
        # recalculate aspect ratio for each image in case image sizes are different
        if aspect == None:
            a = (img_height/img.shape[0])/(img_width/img.shape[1])

        plot_image(img=img, img_output_folder=img_output_folder, title=title, default_title=default_title, \
                   title_fontsize=title_fontsize, cmap=cmap, aspect=a, save_imgs=save_imgs, thre=thre, \
                   log_scale = False, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks, \
                   interpolation=interpolation, h=h, w=w)


# ===========================================================================================
# ratio images
# ===========================================================================================

def ratio_images(pixels, metadata=None, idxs = [1,2], normalize = None, handle_infinity = 'maximum', titles = None, \
                aspect = None, scale = .999, save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                log_scale = False, threshold = None, title_fontsize = 10, image_savetype = 'figure', \
                axis_tick_marks = False, interpolation = 'none', h = 6, w = 6):
    """
    Generates ratio images from the given pixels, metadata, and pair of indices.
    Each image is divided by the other to get the ratio images.
    
    Args:
        pixels (list or array): 
            The images to be displayed.
        metadata (dict): 
            The metadata for the images. This should include the mass list and the image dimensions.
        idxs (list):
            The indices of the images to be used. len must be 2.
        normalize (str): 
            The normalization method. Options are 'None', or 'base_peak'.
            'base_peak' will normalize the images to the base peak intensity before division.
        handle_infinity (str):
            The method to handle infinity values. Options are 'maximum', 'infinity', or 'zero'.
            'maximum' will set the infinity values to the maximum value in the image.
            'infinity' will set the infinity values to infinity.
            'zero' will set the infinity values to zero.
        titles (list):
            The titles for the images. If None, the titles will be determined based on the mass list.
        aspect (float):
            The aspect ratio of each pixel for display. If None, the aspect ratio will be calculated based on the image dimensions.
        scale (float):
            The quantile to lower intensity values to. Default is .999.
            Any pixel with an intensity greater than the pixel with this quantile will be decreased to this intensity. 
            This is done to prevent saturation of the color scale.
            Ignored if threshold is given.
        save_imgs (bool):
            If True, the images will be saved to the MSI_data_output directory.
        MSI_data_output (str):
            The directory to save the images to. If None, the images will be saved to the current working directory.
        cmap (str):
            The colormap to use for the images. Default is 'viridis'.
        log_scale (bool):
            If True, the images will be displayed on a log scale.
            If False, the images will be displayed on a linear scale.
        threshold (float):
            The threshold for the images. Any pixel with an intensity greater than the threshold will be decreased to the threshold.
            If None, the threshold will be determined based on the scale.
        title_fontsize (int):
            The font size of the titles. Default is 10.
        image_savetype (str):
            The type of image to save. Options are 'figure', 'image', or 'array'.
            'figure' will save the image as a figure with a colorbar and title.
            'image' will save the image as an image without a colorbar or title.
            'array' will save the image as an array in csv format.
        axis_tick_marks (bool):
            If True, the axis tick marks will be shown. Default is False.
        interpolation (str):
            The interpolation method to use for displaying the images. Default is 'none'.
            Using 'nearest' or 'none' will make the images look pixelated, while 'bilinear' will make them look smoother/blurrier.
            See https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html for more options.
    """
    
    ratio_imgs = get_ratio_imgs(pixels, metadata, idxs, normalize, handle_infinity, titles)
    display_ratio_images(ratio_imgs, metadata, titles, aspect, scale, save_imgs, MSI_data_output, cmap, \
                         log_scale, threshold, title_fontsize, idxs, image_savetype=image_savetype, \
                         axis_tick_marks=axis_tick_marks, interpolation=interpolation, h=h, w=w)

def get_ratio_imgs(pixels, metadata=None, idxs = [1,2], normalize = None, handle_infinity = 'maximum', titles = None):
    assert handle_infinity.lower() in ['maximum', 'infinity', 'zero'], "handle_infinity must be in ['maximum', 'infinity', 'zero']"
    """
    Normalizes pixels before getting ratio images. 
    Each image is divided by the other to get the ratio image.
    If the images are of varying size, the images are resized to the size of the image that is being divided by the other.

    Returns:
        ratio_imgs (list): The ratio images.
    """

    idxs = idxs[:2]
    normalize = get_normalize_value(normalize, possible_entries=['None', 'base_peak'])

    imgs = [pixels[i] for i in idxs]

    # Ensure images are all the same size
    shapes = [img.shape for img in imgs]
    idxs_to_reshape = np.any(~np.equal(shapes[0],shapes), axis = 1)
    for idx, i in enumerate(idxs_to_reshape):
        if i: imgs[idx] = resize(imgs[idx], shapes[0], order=0)

    if normalize == "base_peak":
        imgs = base_peak_normalize_pixels(imgs)

    ratio_imgs = []

    # Get default values for where the images are both 0
    img_background = np.zeros(imgs[0].shape)
    img_background[np.where((imgs[0] == 0) & (imgs[1] == 0))] = 1
    img1_background = img_background.copy()
    img2_background = img_background.copy()

    # set locations where the image you are dividing by is 0 to inf or zero
    if handle_infinity in ['maximum', 'infinity']: fill_val = np.inf
    else: fill_val = 0
    img1_background[np.where((imgs[0]!=0)&(imgs[1]==0))] == fill_val
    img2_background[np.where((imgs[0]==0)&(imgs[1]!=0))] == fill_val

    # get the ratio images
    ratio_imgs.append(np.divide(imgs[0], imgs[1], out=img1_background, where=imgs[1]!=0))
    ratio_imgs.append(np.divide(imgs[1], imgs[0], out=img2_background, where=imgs[0]!=0))

    if handle_infinity == 'maximum':
        # set locations where you divided a non-zero value by zero to the maximum
        ratio_imgs[0][np.isinf(ratio_imgs[0])] = ratio_imgs[0][~np.isinf(ratio_imgs[0])].max()
        ratio_imgs[1][np.isinf(ratio_imgs[0])] = ratio_imgs[1][~np.isinf(ratio_imgs[1])].max()

    return ratio_imgs

def display_ratio_images(ratio_imgs, metadata=None, titles = None, aspect = None, scale = .999,save_imgs = False, \
                         MSI_data_output = None, cmap = 'viridis', log_scale = False, threshold = None, \
                         title_fontsize = 10, idxs = [1,2], image_savetype = 'figure', axis_tick_marks=False, \
                         interpolation='none', h=6, w=6):    
    """
    Displays the fractional abundance images in the fract_imgs array.
    """
    if metadata == None:
        default_titles = ["Image "+str(i) + "/Image "+str(j) for i,j in zip(idxs, idxs[::-1])]
    else:
        mass_list = metadata["final_mass_list"]
        default_titles = determine_titles(mass_list, idxs = idxs, ratio_img = True)

    # make sure save directory exists
    if MSI_data_output == None:
        MSI_data_output = os.getcwd()
    img_output_folder = os.path.join(MSI_data_output,'ratio_images')
    if save_imgs:
        if not os.path.exists(img_output_folder):
            os.makedirs(img_output_folder)

    # plot each image
    if metadata == None:
        img_height, img_width = 1.0, 1.0
    else:
        img_height, img_width = metadata['image_dimensions']
    # use manually given aspect ratio
    a = aspect

    if threshold:
        thre = threshold

    for i in range(len(ratio_imgs)):
        img = ratio_imgs[i]

        if titles == None:
            title = default_titles[i]
        else:
            title = titles[i]
        default_title = default_titles[i]

        if scale and (not threshold):
            thre = np.quantile(img, scale)
        elif not threshold: 
            thre = img.max()

        # recalculate aspect ratio for each image in case image sizes are different
        if aspect == None:
            a = (img_height/img.shape[0])/(img_width/img.shape[1])

        plot_image(img=img, img_output_folder=img_output_folder, title=title, default_title=default_title, \
                   title_fontsize=title_fontsize, cmap=cmap, aspect=a, save_imgs=save_imgs, thre=thre, \
                   log_scale=log_scale, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks, \
                   interpolation=interpolation, h=h, w=w)

def plot_image(img, img_output_folder, title, default_title, title_fontsize, cmap, aspect, save_imgs, thre, \
    log_scale = False, image_savetype = 'figure', axis_tick_marks = False, interpolation='none', h = 6, w = 6):

    """
    The function that handles plotting the images for each display function.
    
    Args:
        img (array):
            The image to be displayed.
        img_output_folder (str):
            The directory to save the images to.
        title (str):
            The title for the image.
        default_title (str):
            The default title for the image, used if the given title causes an error when saving.
        title_fontsize (int):
            The font size of the title.
        cmap (str):
            The colormap to use for the image.
        aspect (float):
            The aspect ratio of each pixel for display.
        save_imgs (bool):
            If True, the image will be saved to the img_output_folder directory.
        thre (float):
            The threshold for the image. Any pixel with an intensity greater than the threshold will be decreased to the threshold.
        log_scale (bool):
            If True, the image will be displayed on a log scale.
            If False, the image will be displayed on a linear scale.
        image_savetype (str):
            The type of image to save. Options are 'figure', 'image', or 'array'.
            'figure' will save the image as a figure with a colorbar and title.
            'image' will save the image as an image without a colorbar or title.
            'array' will save the image as an array in csv format.
        axis_tick_marks (bool):
            If True, the axis tick marks will be shown. Default is False.
        interpolation (str):
            The interpolation method to use for displaying the image. Default is 'none'.
            Using 'nearest' or 'none' will make the image look pixelated, while 'bilinear' will make it look smoother/blurrier.
            See https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html for more options.
        h (int):
            The height of the figure in inches. Only used if image_savetype is 'figure'.
        w (int):
            The width of the figure in inches. Only used if image_savetype is 'figure
    """

    # Save images as publication-style figure, including a colorbar and title
    if image_savetype == 'figure':
        #TODO: Allow for variable figsize
        plt.figure(figsize=(w,h))
        if log_scale:
            # Prevent -inf values from taking log of zero
            min_thre = np.min(img[np.nonzero(img)])/10
            min_thre_img = np.where(img==0, min_thre, img)
            plt.imshow(min_thre_img, cmap = cmap, aspect = aspect, norm = colors.LogNorm(), interpolation=interpolation)
        else:
            plt.imshow(img, cmap = cmap, aspect = aspect, vmin = 0, vmax=thre, interpolation=interpolation)

        plt.title(title, fontsize = title_fontsize)
        
        if not axis_tick_marks:
            plt.xticks([])
            plt.yticks([])

        plt.colorbar()

        if save_imgs: 
            try:
                plt.savefig(os.path.join(img_output_folder,title.replace(':','_').replace('\n',' ').replace('>','').replace('/','')+'.png') )
            except:
                plt.savefig(os.path.join(img_output_folder,default_title.replace(':','_').replace('\n',' ').replace('>','').replace('/','')+'.png') )
        else:
            plt.show()
        plt.close()
        plt.clf()

    # Save as an image without any colorbar or title
    elif image_savetype == 'image':
        cm = plt.get_cmap(cmap)

        if log_scale:
            # Prevent -inf values from taking log of zero
            min_thre = np.min(img[np.nonzero(img)])/10
            min_thre_img = np.where(img==0, min_thre, img)
            img = np.log10(min_thre_img)
            thre = np.log10(thre)

        img = np.where(img>thre, thre, img)
        
        # prevent division by zero
        if img.max()-img.min() == 0:
            normed_img = np.zeros_like(img)
        else:
            normed_img = (img-img.min())/(img.max()-img.min())

        colored_img = cm(normed_img)
        colored_img = (colored_img[:,:,:3]*255).astype(np.uint8)
        
        # get dimensions for resizing
        h, w = colored_img.shape[:2]
        
        pil_img = Image.fromarray(colored_img)
        if aspect >=1:
            pil_img = pil_img.resize((w, round(h*aspect)), resample=0)
        else:
            pil_img = pil_img.resize((w//aspect, h), resample=0)

        if save_imgs:
            try:
                pil_img.save(os.path.join(img_output_folder,title.replace(':','_').replace('\n',' ').replace('>','').replace('/','')+"_threshold-"+str(thre)+'.png') )
            except:
                pil_img.save(os.path.join(img_output_folder,default_title.replace(':','_').replace('\n',' ').replace('>','').replace('/','')+"_threshold-"+str(thre)+'.png') )
        else:
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(pil_img)
            plt.show()
            plt.clf()

    # Save as an array in csv format
    elif image_savetype == 'array':
        if log_scale:
            # Prevent -inf values from taking log of zero
            min_thre = np.min(img[np.nonzero(img)])/10
            min_thre_img = np.where(img==0, min_thre, img)
            img = np.log10(min_thre_img)
            thre = np.log10(thre)
        img = np.where(img>thre, thre, img)
        if save_imgs: 
            try:
                np.savetxt(os.path.join(img_output_folder,title.replace(':','_').replace('\n',' ').replace('>','').replace('/','')+"_threshold-"+str(thre)+'.csv'), img, delimiter=",")
            except:
                np.savetxt(os.path.join(img_output_folder,default_title.replace(':','_').replace('\n',' ').replace('>','').replace('/','')+"_threshold-"+str(thre)+'.csv'), img, delimiter=",")