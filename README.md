# MSIGen
MSIGen is designed for converting mass spectrometry imaging (MSI) data from the raw line-scan data to a visualizable format and is designed with nano-DESI MSI in mind. It has premade files for converting to images using a GUI, jupyter notebook, or from the command line.

## Installation on Windows
Using Anaconda (https://www.anaconda.com/download), create a new environment titled "MSIGen" with python >=3.8 and activate it. Then, MSIGen can be installed using the pip package manager.

Run the following in anaconda prompt:
```
conda create --name MSIGen python=3.9
conda activate MSIGen
pip install MSIGen
```
### For GUI tool:
Download "make GUI shortcut.py" from the tests folder in the Github repository. Run this code from Anaconda prompt.
```
conda activate MSIGen
python "C:/path/to/make GUI shortcut.py"
```
After running with the actual location of "make GUI shortcut.py", there should be a shortcut called "MSIGen GUI" on your desktop. This runs the GUI for MSIGen.

### For Jupyter Notebook Tool:
Download "MSIGen_jupyter.ipynb" from the tests folder in the Github repository. Open Anaconda and run Jupyter Notebook in the MSIGen environment. Open "MSIGen_jupyter.ipynb" from Jupyter Notebook.

### For Command Line Interface Tool:
Download "MSIGen_CLI.py" from the tests folder in the Github repository. Create a configuration file for your experiment. An example can be found in the tests folder. Run the following in Anaconda Prompt:
```
conda activate MSIGen
python "C:/Path/to/MSIGen_CLI.py" "C:/path/to/config_file1.json" "C:/path/to/config_file2.json"
```
Supply one configuration file for each dataset to be processed.


