# MSIGen
MSIGen is designed for converting mass spectrometry imaging (MSI) data from the raw line-scan data to a visualizable format and is designed with nano-DESI MSI in mind. It has premade files for converting to images using a GUI, jupyter notebook, or from the command line.

## Installation on Windows
Using Anaconda (https://www.anaconda.com/download), create a new environment titled "MSIGen" with python >=3.8 and activate it. Then, MSIGen can be installed using the pip package manager.

from CLI:
    $conda create --name MSIGen python=3.9
    $conda activate MSIGen
    $pip install MSIGen

### For gui:
    Locate the package and make a shortcut to "MSIGen GUI.bat" located under "GUI Shortcuts". Alternatively, download the folder and make a shortcut to those.

### For jupyter notebook:
    Locate the 