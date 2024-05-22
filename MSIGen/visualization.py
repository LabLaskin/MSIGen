import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize
import os
from PIL import Image


# ===========================================================================================
# Raw or normalized image visualization
# ===========================================================================================

def get_normalize_value(normalize, possible_entries = ['None','TIC','intl_std']):
    """Parses the value of the normalize variable. Allows for error handling and for some leeway in mistyping the keywords."""
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

def match_to_mass_list(mass_list, std_idx = None, std_precursor = None, std_mass = None, std_fragment = None, std_mobility = None, std_charge = None):
    # use given std_idx if possible
    if std_idx != None:
        return std_idx

    # determine what mass arrays should look like based on given values
    elif any(np.array((std_precursor, std_mass, std_fragment, std_mobility, std_charge))!=None):
        comparison_arrays = []
        
        # if the value has both std_precursor and std_mass or std_precursor and std_fragment, check only for ms2 values
        if all(np.array((std_precursor, std_mass))!=None):
            comparison_arrays.append(np.array([std_precursor, std_mass, std_mobility, std_charge]))
        elif all(np.array((std_precursor, std_fragment))!=None):
            comparison_arrays.append(np.array([std_precursor, std_mass, std_mobility, std_charge]))
        
        # otherwise check for ms2 or ms1 scans
        else:
            if std_mass == None and std_fragment!=None:
                std_mass = std_fragment
            elif std_mass == None and std_precursor!=None:
                std_mass = std_precursor

            comparison_arrays.append(np.array([std_precursor, std_mass, std_mobility, std_charge]))
            comparison_arrays.append(np.array([std_mass, std_mobility, std_charge]))

        for i, mass in mass_list:
            matches = np.zeros(len(mass_list))

            for comparison_mass in comparison_arrays:
                if len(mass) == len(comparison_mass):
                    matches[i] = np.sum(np.array(mass)==comparison_mass)
                    break
                else:
                    matches[i] = 0 
        
        num_matches = np.max(matches)
        best_match = np.argwhere(matches == np.max(matches))
        
        assert num_matches == 0, 'There were no matches to the given mass, mobility, or charge values'
        assert best_match.shape != 1, 'There were multiple entries on your mass list that matched the given mass, mobility, or charge values'
        
        std_idx = best_match.item()
        return std_idx

    else:
        assert any(np.array((std_idx, std_precursor, std_mass, std_fragment, std_mobility, std_charge))!=None), "At lease one of the followind kwargs must be defined: \
        std_idx, std_precursor, std_mass, std_fragment, std_mobility, std_charge"

def normalize_pixels(pixels, std_idx):
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

            pixels_normed.append(np.divide(i, std_img_tmp, out=np.zeros_like(i), where=std_img_tmp!=0))
    
    # If the pixels are in an array, normalize them all together
    elif type(pixels) == type(np.array(0)):
        pixels_normed = np.divide(pixels, pixels[std_idx], out=np.zeros_like(pixels), where=pixels[std_idx]!=0)
    
    return pixels_normed

def base_peak_normalize_pixels(pixels):
    for i, img in enumerate(pixels):
        if img.max():
            pixels[i]=img/img.max()
    return pixels

def get_and_display_images(pixels, metadata, normalize = None, std_idx = None, std_precursor = None, std_mass = None, \
                        std_fragment = None, std_mobility = None, std_charge = None, aspect = None, scale = .999, \
                        how_many_images_to_display = 'all', save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                        titles = None, threshold = None, title_fontsize = 10, image_savetype = "figure", axis_tick_marks = False):
    pixels_normed = get_pixels_to_display(pixels, metadata, normalize, std_idx, std_precursor, std_mass, std_fragment, std_mobility, std_charge)
    display_images(pixels_normed, metadata, aspect, scale, how_many_images_to_display, save_imgs, MSI_data_output, cmap, titles, threshold, \
                   title_fontsize, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks)

def get_pixels_to_display(pixels, metadata, normalize = None, std_idx = None, std_precursor = None, std_mass = None, std_fragment = None, std_mobility = None, std_charge = None):
    """Normalizes MS1 pixels to TIC or to an internal standard.
    The if images are of varying size, the standard image is reshaped to the size of the image to be normalized."""      

    normalize = get_normalize_value(normalize, ['None', 'TIC', 'intl_std'])

    mass_list = metadata["final_mass_list"]

    if normalize == 'intl_std':
        # find the index of the standard
        std_idx = match_to_mass_list(mass_list, std_idx, std_precursor, std_mass, std_fragment, std_mobility, std_charge)

        pixels_normed = normalize_pixels(pixels, std_idx)

    elif normalize == 'TIC':
        pixels_normed = normalize_pixels(pixels, 0)

    else:
        pixels_normed = pixels

    return pixels_normed

def display_images(pixels_normed, metadata, aspect = None, scale = .999, how_many_images_to_display = 'all', \
                    save_imgs = False, MSI_data_output = None, cmap = 'viridis', titles = None, threshold = None, \
                    title_fontsize = 10, image_savetype = "figure", axis_tick_marks = False):

    # parse args
    if how_many_images_to_display == 'all':
        how_many_images_to_display = len(pixels_normed)
    if type(how_many_images_to_display) in [str, int, float]:
        try:
            how_many_images_to_display = list(range(int(how_many_images_to_display)))
        except:
            raise TypeError("how_many_images_to_display must be either 'all' or an integer")
    if type(how_many_images_to_display) in [list, tuple]:
        try:
            how_many_images_to_display = [int(i) for i in how_many_images_to_display]
        except:
            raise TypeError("how_many_images_to_display must be 'all', an integer, or a list of integers")

    # Get the titles for all figures:
    mass_list = metadata["final_mass_list"]
    
    default_titles = determine_titles(mass_list, idxs = how_many_images_to_display)

    # make sure save directory exists
    if MSI_data_output == None:
        MSI_data_output = os.getcwd()
    img_output_folder = os.path.join(MSI_data_output,'images')
    if save_imgs:
        if not os.path.exists(img_output_folder):
            os.makedirs(img_output_folder)

    # plot each image
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
                cmap=cmap, aspect=a, save_imgs=save_imgs, thre=thre, log_scale = False, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks)


def determine_titles(mass_list, idxs = None, fract_abund = False, ratio_img=False):
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

def fractional_abundance_images(pixels, metadata, idxs = [1,2], normalize = None,titles = None, \
                        aspect = None, save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                        title_fontsize = 10, image_savetype = 'figure', scale = 1.0, threshold=None, \
                        axis_tick_marks=False):
    
    fract_imgs = get_fractional_abundance_imgs(pixels, metadata, idxs, normalize)
    display_fractional_images(fract_imgs, metadata, titles, aspect, save_imgs, MSI_data_output, cmap, \
                              title_fontsize, idxs, image_savetype=image_savetype, scale=scale, \
                              threshold=threshold, axis_tick_marks=axis_tick_marks)

def get_fractional_abundance_imgs(pixels, metadata, idxs = [1,2], normalize = None):
    normalize = get_normalize_value(normalize, ['None', 'base_peak'])

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

def display_fractional_images(fract_imgs, metadata, titles = None, aspect = None,\
                            save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                            title_fontsize = 10, idxs = [1,2], image_savetype='figure', \
                            scale = 1.0, threshold = None, axis_tick_marks=False):    

    mass_list = metadata["final_mass_list"]
    default_titles = determine_titles(mass_list, idxs = idxs, fract_abund=True)

    # make sure save directory exists
    if MSI_data_output == None:
        MSI_data_output = os.getcwd()
    img_output_folder = os.path.join(MSI_data_output,'images')
    if save_imgs:
        if not os.path.exists(img_output_folder):
            os.makedirs(img_output_folder)

    if threshold:
        thre = threshold

    # plot each image
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

        plot_image(img=img, img_output_folder=img_output_folder, title=title, default_title=default_title, title_fontsize=title_fontsize, \
                   cmap=cmap, aspect=a, save_imgs=save_imgs, thre=thre, log_scale = False, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks)


# ===========================================================================================
# ratio images
# ===========================================================================================

def ratio_images(pixels, metadata, idxs = [1,2], normalize = None, handle_infinity = 'maximum', titles = None, \
                aspect = None, scale = .999,save_imgs = False, MSI_data_output = None, cmap = 'viridis', \
                log_scale = False, threshold = None, title_fontsize = 10, image_savetype = 'figure',axis_tick_marks=False):
    
    ratio_imgs = get_ratio_imgs(pixels, metadata, idxs, normalize, handle_infinity, titles)
    display_ratio_images(ratio_imgs, metadata, titles, aspect, scale, save_imgs, MSI_data_output, cmap, log_scale, \
                         threshold, title_fontsize, idxs, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks)

def get_ratio_imgs(pixels, metadata, idxs = [1,2], normalize = None, handle_infinity = 'maximum', titles = None):
    assert handle_infinity.lower() in ['maximum', 'infinity', 'zero'], "handle_infinity must be in ['maximum', 'infinity', 'zero']"

    idxs = idxs[:2]
    normalize = get_normalize_value(normalize, ['None', 'base_peak'])

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

def display_ratio_images(ratio_imgs, metadata, titles = None, aspect = None, scale = .999,save_imgs = False, \
                         MSI_data_output = None, cmap = 'viridis', log_scale = False, threshold = None, \
                         title_fontsize = 10, idxs = [1,2], image_savetype = 'figure', axis_tick_marks=False):    

    mass_list = metadata["final_mass_list"]
    default_titles = determine_titles(mass_list, idxs = idxs, ratio_img = True)

    # make sure save directory exists
    if MSI_data_output == None:
        MSI_data_output = os.getcwd()
    img_output_folder = os.path.join(MSI_data_output,'images')
    if save_imgs:
        if not os.path.exists(img_output_folder):
            os.makedirs(img_output_folder)

    # plot each image
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

        plot_image(img=img, img_output_folder=img_output_folder, title=title, default_title=default_title, title_fontsize=title_fontsize, \
                   cmap=cmap, aspect=a, save_imgs=save_imgs, thre=thre, log_scale = log_scale, image_savetype=image_savetype, axis_tick_marks=axis_tick_marks)

def plot_image(img, img_output_folder, title, default_title, title_fontsize, cmap, aspect, save_imgs, thre, \
    log_scale = False, image_savetype='figure', axis_tick_marks = False):

    # Save images as publication-style figure, including a colorbar and title
    if image_savetype == 'figure':
        plt.figure(figsize=(6,6))
        if log_scale:
            # Prevent -inf values from taking log of zero
            min_thre = np.min(img[np.nonzero(img)])/10
            min_thre_img = np.where(img==0, min_thre, img)
            plt.imshow(min_thre_img, cmap = cmap, aspect = aspect, norm = colors.LogNorm(), interpolation='none')
        else:
            plt.imshow(img, cmap = cmap, aspect = aspect, vmin = 0, vmax=thre, interpolation='none')

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
            
