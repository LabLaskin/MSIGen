"""This module provides a function to subclass the base MSIGen object for different file formats."""

import importlib
import os

class msigen(object):
    """
    This function subclasses the base MSIGen object for different file formats.
    
    Parameters:
        example_file (str or list): The file path or list of file paths to be processed.
        Parameters are passed to MSIGen.base_class.MSIGen_base.
    
    Returns:
        An instance of the appropriate class based on the file format.
    """

    def __new__(cls, *args, **kwargs):
        if "example_file" in kwargs:
            example_file = kwargs["example_file"]
        elif len(args) > 0:
            example_file = args[1]
        else:
            example_file = None

        # Initialize the base class without data files if example_file is None
        if example_file is None:
            module = importlib.import_module('MSIGen.base_class')
            return module.MSIGen_base(*args, **kwargs)
        
        # Check the file extension of the example_file and load the appropriate module
        if type(example_file) == str:
            file_extension = os.path.splitext(example_file)[1].lower()
        if type(example_file) in [list, tuple]:
            file_extension = os.path.splitext(example_file[0])[-1].lower()
            
        if file_extension == ".d":  # Customize extension matching as needed
            module = importlib.import_module('MSIGen.D')
            return module.MSIGen_D(*args, **kwargs)
        elif file_extension == ".raw":
            module = importlib.import_module('MSIGen.raw')
            return module.MSIGen_raw(*args, **kwargs)
        elif file_extension == ".mzml":
            module = importlib.import_module('MSIGen.mzml')
            return module.MSIGen_mzml(*args, **kwargs)
        else:
            raise ValueError(f"Invalid file extension{file_extension}. Supported file extensions are: '.d', '.mzml', '.raw'")

    @staticmethod
    def load_pixels(path=None):
        """
        This function loads pixel data from the specified file without initilizing the class beforehand.
        
        Parameters:
            path (str): 
                The file path to load pixel data from. 
                If path is None, the current directory will be searched for a file named pixels.npy, pixels.npz, or pixels.csv and this file will be loaded.

        Returns:
            Pixel data loaded from the file.
        """
        # Load the base class module and call the load_pixels method
        module = importlib.import_module('MSIGen.base_class')
        return module.MSIGen_base().load_pixels(path)

    @classmethod
    def get_metadata_and_params(cls, *args, **kwargs):
        """
        This is an alias for the __new__ method to allow for compatibility with older versions of MSIGen_jupyter files.ipynb files.
        """
        return cls.__new__(cls, *args, **kwargs)