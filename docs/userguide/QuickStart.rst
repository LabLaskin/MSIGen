Quick Start for MSIGen
===================================

MSIGen is most easily used through Jupyter Notebooks or through the GUI. This guide will show an example of how to use MSIGen in a Jupyter Notebook.

First, ensure you have installed MSIGen and Jupyter Notebook as described in the `Installation guide <Installation.html>`_.

Next, download the example Jupyter Notebook file from the GitHub repository (https://github.com/LabLaskin/MSIGen/blob/main/other_files/MSIGen_jupyter.ipynb) and open it in Jupyter Notebook or VSCode.

Processing Your Data
---------------------
The first cell of the notebook contains the parameters used to process the data from line scans into a NumPy array.

.. code:: python

    # This should be a string or list of strings.
    # Using a single file path will use all files that match the file name except for the final number
    example_file = r"C:\example_data\testline1.d"

    # excel or csv file containing your mass list
    mass_list_dir = r"C:\example_data\testmasslist.xlsx"

    # Do not comment these out. If not used, they will be ignored
    mass_tolerance_MS1 = 10
    mass_tolerance_MS1_units = 'ppm'    #ppm or mz
    mass_tolerance_prec = 1
    mass_tolerance_prec_units = 'mz'    #ppm or mz
    mass_tolerance_frag = 10
    mass_tolerance_frag_units = 'ppm'   #ppm or mz
    mobility_tolerance = .1            
    mobility_tolerance_units = '1/K0'   #'μs' or '1/K0', whichever us used in the mobility list

    # the dimensions of the scanned area
    img_height = 10
    img_width = 10
    image_dimensions_units = 'mm'

    # whether the files contain MS2 or mobility data
    is_MS2 = False
    is_mobility = False

    # True if you want all images to be the same size regardless of differences in number of scans (generally True)
    normalize_img_sizes = True

    # Use the max, min, or mean number of pixels per line for the images. Can instead use an integer to manually set this value. 
    pixels_per_line = "mean"

    # Where you want to save your images numpy array file
    output_file_loc = r"C:\example_data"

    # =====================================================================================
    from MSIGen import msigen
    MSIGen_generator = msigen(example_file=example_file, mass_list_dir=mass_list_dir, tol_MS1=mass_tolerance_MS1, \
        tol_MS1_u=mass_tolerance_MS1_units, tol_prec=mass_tolerance_prec, tol_prec_u=mass_tolerance_prec_units, \
        tol_frag=mass_tolerance_frag, tol_frag_u=mass_tolerance_frag_units, tol_mob=mobility_tolerance, \
        tol_mob_u=mobility_tolerance_units, h=img_height, w=img_width, hw_units=image_dimensions_units, \
        is_MS2=is_MS2, is_mobility=is_mobility, normalize_img_sizes=normalize_img_sizes, \
        pixels_per_line=pixels_per_line, output_file_loc=output_file_loc, in_jupyter = True, testing = False)

Here is a short explanation of each parameter:

1. example_file: The path of one of your line scan files. e.g. "C:/path/to/your/file.d". Include an r in front of the string when using backslashes to ensure they are interpreted correctly. All other line scan files in the folder with the same name apart from a line number directly before the file extention will be included in the processing. Accepted file formats include Agilent .d, Bruker .d (.tsf, .baf, .tdf formats), Thermo .raw, and open-source .mzML files. If your data is not in one of these formats, please see the `Converting Line Scan Data to .mzML Format`_ for instructions on converting your data to .mzML format using ProteoWizard's MSConvert tool.

2. mass_list_dir: The file path of your mass list file. e.g. "C:/path/to/your/masslist.xlsx". This file can be either an Excel or csv file. The mass list file can also contain precursor `m/z`, fragment `m/z`, and/or ion mobility values. An example mass list file is included in the GitHub repository (https://github.com/LabLaskin/MSIGen/blob/main/other_files/example_mass_list.xlsx).

3. mass_tolerance_MS1, mass_tolerance_prec, mass_tolerance_frag, mobility_tolerance: The tolerance and units for matching `m/z` values in MS1 spectra, for precursor and fragment ions in MS2 spectra, and for mobility values if ion mobiility is used. The units can be either 'ppm' or 'mz' for mass tolerances and μs or 1/K0 for mobility tolerances.

4. img_height, img_width, image_dimensions_units: The height and width of the scanned area along with the units used (e.g. 'mm').

5. normalize_img_sizes: Whether to make all images the same size regardless of differences in number of scans (generally True).

6. pixels_per_line: Use the max, min, or mean number of pixels (spectra) per line for the images. Can instead use an integer to manually set this value.

7. output_file_loc: The path to the directory where the output files will be saved.

After setting the parameters to fit your data, run the cell. Then run the following cell to process your data:

.. code:: python

    metadata, pixels = MSIGen_generator.get_image_data(verbose=True)

This may take a few seconds or minutes depending on the size of your data. A NumPy array file (pixels.npy) and a JSON metadata file (pixels_metadata.json) will be saved in the output directory you specified.

Alternatively, you can instead run the next cell to load a previously saved NumPy array and metadata file saved at load_path:

.. code:: python

    from MSIGen import msigen
    load_path = r"C:\example_data\pixels.npy"   # set this to '' to check for pixels.npy file in your current directory
    pixels, metadata = msigen.load_pixels(load_path)

At this point you have processed all of your data, creating a numpy array of shape (n+1, y, x) where n is the number of masses included in the mass list, plus one for the TIC image at the beginning. y and x are the height and width of the image in pixels, respectively. You can now visualize your data.



Converting Line Scan Data to .mzML Format
------------------------------------------

If your data is not in one of the supported vendor formats (Agilent .d, Bruker .d (.tsf, .baf, .tdf formats), or Thermo .raw), you can convert it to the open-source .mzML format using ProteoWizard's MSConvert tool. You can download ProteoWizard from https://proteowizard.sourceforge.io/download.html. After installing it, open MSConvert GUI and select the files you want to convert by clicking "Add Files". Then select "mzML" as the output format and choose an output folder. Finally, click "Start" to convert your files.


Visualizing Your Data
----------------------

After processing your data and obtaining the NumPy array and metadata, you can visualize your data as ion images using the following cell in the Jupyter Notebook:

.. code:: python

    from MSIGen import visualization as vis

    # Sets the maximum pixel intensity to the this quantile (or use threshold)
    scale = 0.999
    # Set the maximum intensity threshold manually (None if using scale instead)
    threshold = None  

    # override automatically calculated aspect ratio (None to use automatic)
    aspect = None

    normalize = 'none'        # Can be 'TIC' 'intl_std' or 'none'

    # ---- Use this if normalize = 'intl_std' ---- #
    # std_idx is 0 for TIC and 1 for the first mass on your mass list.
    std_idx = 1   # Uses this value by default

    # specify these values and set std_idx = None if you do not know the index of the standard
    # Ignore these if std_idx is not None.
    std_precursor = None
    std_mass = None   
    std_fragment = None
    std_mobility = None
    std_charge = None

    # True or False
    save_imgs = False

    # save images as publication-style figures, just the images, or as image arrays in .csv format
    image_savetype = 'figure'   # "figure", "image", "array"

    # path to save output images to. Use file path instead if it is different than output_file_loc
    MSI_data_output = output_file_loc

    # 'all' to display all images. Use an integer to display only that many images
    # Use a list of integers to specify which images you would like to save.
    how_many_images_to_display = 'all'

    # Colormap for images
    cmap = 'viridis'

    # whether to smooth images with interpolation. 
    # 'none' for no interpolation, None or 'linear' for interpolation
    interpolation='none'

    # None or a list of titles if you want to override the default titles
    titles = None
    title_fontsize = 10

    # whether to display tick marks on the edges of the image
    axis_tick_marks = False

    # height and width of the images in inches for publication-style figures (image_savetype = 'figure')
    h, w = 6, 6

    # ---- Normalizes images, displays, and saves images if desired ---- #
    vis.get_and_display_images(pixels, metadata, normalize, std_idx, std_precursor, std_mass, std_fragment, \
                            std_mobility, std_charge, aspect, scale, how_many_images_to_display, save_imgs, \
                            MSI_data_output, cmap, titles, threshold, title_fontsize=title_fontsize, \
                            axis_tick_marks=axis_tick_marks, image_savetype=image_savetype, \
                            interpolation=interpolation, h=h, w=w)

Here is a short explanation of each parameter:

1. scale, threshold: Scale sets the maximum pixel intensity to the the pixel with the intensity of this quantile (eg. 0.999 sets the maximum intensity to the pixels brighter than 99.9% of pixels) for displaying the images. If threshold is not None, scale will be ignored and the maximum pixel intensity will be a equal to threshold.

2. aspect: Defines the aspect ratio of each pixel in the image, since pixels are not generally square when using most continuous line-wise acquisition methods. If this is None, the aspect ratio will be automatically determined from the img_height and img_width parameters used when processing the data.

3. normalize: Can be 'TIC', 'intl_std', or 'none' to normalize the images to total ion current, an internal standard, or not at all, respectively.

4. std_idx: If normalizing to an internal standard, this is the index of the standard in the mass list. This is 0 for TIC and 1 for the first mass on your mass list. If you do not know the index of the standard, check your mass list or set this to None and instead specify the precursor m/z, fragment m/z, mobility value, and charge of the standard using std_precursor, std_mass, std_fragment, std_mobility, and std_charge parameters.

5. save_imgs: Whether to save the images to files (True or False).

6. image_savetype: Whether to save images as publication-style figures (with a title and colorbar), just the images, or as image arrays in .csv format. Options are "figure", "image", or "array".

7. MSI_data_output: The path to the directory where the output images will be saved.

8. how_many_images_to_display: 'all' to display all images. Use an integer to display only that many images, starting from the TIC then the next images in the list. Use a list of integers that correspond to the images' locations in the mass list to specify which images you would like to save.

9. cmap: The colormap used for the images. This can be any valid matplotlib colormap (e.g. 'viridis', 'plasma', 'magma', 'cividis', 'turbo', etc.).

10. interpolation: Whether to smooth images with interpolation. 'none' for no interpolation, None or 'linear' for linear interpolation.

11. titles: None or a list of titles if you want to override the default titles.

12. title_fontsize: The font size of the titles if using publication-style figures.

13. axis_tick_marks: Whether to display tick marks on the edges of the image (True or False).

14. h, w: The height and width of the images in inches for publication-style figures.

After setting the parameters to fit your preferences, run the cell. The images will be displayed in the notebook if save_imgs = False or saved to files if you save_imgs = True.

Ratio Images and Fractional Abundance Images
----------------------------------------------------------------------

MSIGen also supports visualizing ratio images and fractional abundance images.
A fractional abundance image is the intensity of a mass divided by the sum of the intensities of a multiple of masses at each pixel, while a ratio image is the ratio of the intensities of two masses at each pixel.
To visualize fractional images, use the following cell in the Jupyter Notebook:

.. code:: python

    from MSIGen import visualization as vis

    # Should be a list of mass list entry indices
    idxs_of_images_to_compare = [1,2]

    # Sets the maximum pixel intensity to the this quantile (or use threshold)
    scale = 1
    # Set the maximum intensity threshold manually (None if using scale instead)
    threshold = 1  

    # override automatically calculated aspect ratio (None to use automatic)
    aspect = None

    # normalize images to their base peak before determining fraction 
    normalize = 'None'     # None or 'base_peak'

    # True or False
    save_imgs = False

    # save images as publication-style figures, just the images, or as image arrays in .csv format
    image_savetype = 'figure'   # "figure", "image", "array"

    # path to save output images to. Use file path instead if it is different than output_file_loc
    MSI_data_output = output_file_loc

    # Colormap for images
    cmap = 'viridis'

    # whether to smooth images with interpolation. 
    # 'none' for no interpolation, None or 'linear' for interpolation
    interpolation='none'

    # None or a list of titles if you want to override the default titles
    titles = None
    title_fontsize = 10

    # whether to display tick marks on the edges of the image
    axis_tick_marks = False

    # height and width of the images in inches for publication-style figures
    h, w = 6, 6

    # displays and saves images
    vis.fractional_abundance_images(pixels, metadata, idxs=idxs_of_images_to_compare, normalize=normalize, \
        titles=titles, aspect=aspect, save_imgs=save_imgs, MSI_data_output=MSI_data_output, cmap=cmap, \
        title_fontsize=title_fontsize, scale=scale, threshold=threshold, axis_tick_marks=axis_tick_marks, \
        image_savetype=image_savetype, interpolation=interpolation, h=h, w=w)

Here is a short explanation of each parameter that differs from the previous visualization function:

1. idxs_of_images_to_compare: A list of two or more indices of the images you want to use to calculate the ratio or fractional abundance. These indices correspond to the locations of the masses in your mass list, with 0 being the TIC and 1 being the first mass on your mass list.

2. normalize: Whether to normalize the images to their base peak (brightest pixel in the image) before calculating the ratio or fractional abundance. Options are None or 'base_peak'.

To visualize ratio images, use the following cell in the Jupyter Notebook:

.. code:: python

    from MSIGen import visualization as vis

    # Should be a list of two mass list entry indices
    idxs_of_images_to_compare = [1,2]

    # Sets the maximum pixel intensity to the this quantile (or use threshold)
    scale = 0.999
    # Set the maximum intensity threshold manually (None if using scale instead)
    threshold = None  

    # override automatically calculated aspect ratio (None to use automatic)
    aspect = None

    # normalize images to their base peak before determining ratio 
    normalize = 'base_peak'        # None or 'base_peak'

    # What to replace infinity values from divide by zero errors with
    handle_infinity = 'maximum'         # 'maximum', 'infinity', or 'zero'

    # True or False
    save_imgs = False

    # save images as publication-style figures, just the images, or as image arrays in .csv format
    image_savetype = 'figure'   # "figure", "image", "array"

    # path to save output images to. Use file path instead if it is different than output_file_loc
    MSI_data_output = output_file_loc

    # Colormap for images
    cmap = 'viridis'

    # whether to smooth images with interpolation. 
    # 'none' for no interpolation, None or 'linear' for interpolation
    interpolation='none'

    # Whether to use a log-scale axis for the colormap
    log_scale = False

    # None or a list of titles for each image if you want to manually create titles
    titles = None
    title_fontsize = 10

    # whether to display tick marks on the edges of the image
    axis_tick_marks = False

    # height and width of the images in inches for publication-style figures
    h, w = 6, 6

    # displays and saves images
    vis.ratio_images(pixels, metadata, idxs=idxs_of_images_to_compare, normalize=normalize, handle_infinity=handle_infinity, titles=titles, \
                    aspect=aspect, scale=scale,save_imgs=save_imgs, MSI_data_output = MSI_data_output, cmap = cmap, log_scale = log_scale, \
                    threshold = threshold, title_fontsize = title_fontsize, axis_tick_marks=axis_tick_marks, image_savetype=image_savetype, \
                    interpolation=interpolation, h=h, w=w)

Here is a short explanation of each parameter that differs from the previous visualization functions:

1. idxs_of_images_to_compare: A list of only two indices corresponding to the images you want to use to calculate the ratio. These indices correspond to the locations of the masses in your mass list, with 0 being the TIC and 1 being the first mass on your mass list.

2. normalize: Whether to normalize the images to their base peak (brightest pixel in the image) before calculating the ratio. Options are None or 'base_peak'.

3. handle_infinity: What to replace infinity values from divide by zero errors with. Options are 'maximum' to replace it with the maximum pixel intensity in the image, 'infinity' to leave it as infinity, or 'zero' to replace it with zero.

4. log_scale: Whether to use a log-scale axis for the colormap.