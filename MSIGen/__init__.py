"""
MSIGen
======

A package for converting spectrometry imaging line scan data files to a visualizable format.
It provides tools for reading, processing, and visualizing mass spectrometry imaging data.

How to use the documentation
----------------------------
Documentation is available as docstrings provided with the code, 
or at `the MSIGen homepage <https://msigen.readthedocs.io>`.

Available subpackages
---------------------
msigen
    This contains a function to initialize the MSIGen object or to load previously saved data.
base_class
    Base class for MSIGen objects, providing common functionality.
D
    Subclass for handling .d file format.
raw
    Subclass for handling .raw file format.
mzml
    Subclass for handling .mzml file format.
visualization
    Tools for visualizing mass spectrometry imaging data.
GUI
    Graphical User Interface for MSIGen, allowing users to interact with the data visually.
tsf
    Tools for handling TSF files from Bruker.

Other attributes
----------------
__version__
    MSIGen version string
__all__
    List of public objects in the MSIGen package.
"""

import os

from MSIGen.msigen import msigen
from MSIGen import visualization
from MSIGen import GUI

try:
    from importlib.metadata import version as get_version, PackageNotFoundError # for python>=3.8, MSIGen requires python>=3.8
except ImportError:
    from importlib_metadata import version as get_version, PackageNotFoundError # for python<3.8

def _get_version():
    try:
        get_version("MSIGen")
        return get_version("MSIGen")
    except PackageNotFoundError:
        # Fallback for when the package is not installed via pip
        try:
            try: 
                import tomllib # for python>=3.11
            except ModuleNotFoundError:
                import pip._vendor.tomli as tomllib # for python>=3.8, <3.11
            base_dir = (os.sep).join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
            pyproject_path = os.path.join(base_dir, "pyproject.toml")
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                return data["tool"]["poetry"]["version"]
        except Exception:
            return "unknown"

__version__ = _get_version()

__all__ = ['msigen', 'base_class', 'D', 'raw', 
           'mzml', 'visualization', 'GUI', 'tsf']


# Remove symbols imported for internal use
del os, PackageNotFoundError, get_version, _get_version