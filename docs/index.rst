.. MSIGen documentation master file, created by
   sphinx-quickstart on Mon Apr 14 12:35:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MSIGen documentation
====================

MSIGen provides tools for processing mass spectrometry imaging data acquired in line-scan mode into images and figures.

This package takes line-scan files as an input and generates a numpy and json file as intermediate outputs containing the image data and metadata, respectively.
MSIGen also includes tools to visualize the generated images as images or figures with a title and colorbar. 
These images can be normalized to TIC, internal standard, or the brightest pixel.
Supported image types are ion images, fractional abundance images, or ratio images.

MSIGen supports multiple vendor and open-source file formats (Agilent .d, Bruker .tsf, .baf, .tdf, Thermo .raw, and open-source .mzML files) and can to handle MS, MS/MS with or without ion mobility data.

Installation instructions can be found on the `GitHub repository <https://github.com/LabLaskin/MSIGen>`_.

If MSIGen was used in your published work, please cite it using the following:
[1.] Hernly E, Hu H, Laskin J. MSIGen: An Open-Source Python Package for Processing and Visualizing Mass Spectrometry Imaging Data. J. Am. Soc. Mass Spectrom. 2024, 35, 10, 2315–2323; doi:10.1021/jasms.4c00178

.. TODO: Add user guides.
.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   userguide/Installation
..   Quickstart
   Processing your Data
   Visualizing your Data
   Examples
   Release Notes

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   MSIGen

