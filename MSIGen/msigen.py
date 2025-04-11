"""This module provides a function to subclass the base MSIGen object for different file formats."""

import importlib
import os

def msigen(example_file, *args, **kwargs):
    file_extension = os.path.splitext(example_file)[1]
    if type(example_file) in [list, tuple]:
        example_file = example_file[0]
        file_extension = os.path.splitext(example_file)[-1].lower()
        
    if file_extension == ".d":  # Customize extension matching as needed
        module = importlib.import_module('D', package='MSIGen')
        return module.MSIGen_D(example_file, *args, **kwargs)
    elif file_extension == ".raw":
        module = importlib.import_module('raw', package='MSIGen')
        return module.MSIGen_raw(example_file, *args, **kwargs)
    elif file_extension == ".mzml":
        module = importlib.import_module('mzml', package='MSIGen')
        return module.MSIGen_mzml(example_file, *args, **kwargs)
    else:
        raise ValueError(f"Invalid file extension{file_extension}. Supported file extensions are: '.d', '.mzml', '.raw'")
