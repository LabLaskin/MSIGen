Installing MSIGen
=================================

Using an environment with python version >=3.9 and <=3.11,
you can install MSIGen using the pip package manager to install from PyPI:

.. code:: bash

   pip install MSIGen

Or, you can install from GitHub repository using pip:

.. code:: bash

   pip install git+https://github.com/LabLaskin/MSIGen.git

If you want to use MSIGen in a Jupyter notebook, you may also need to install jupyter notebook:

.. code:: bash

   pip install notebook

If you are planning on using Bruker .d data in the .baf format, you will also need to install pyBaf2Sql from GitHub:

.. code:: bash

   pip install git+https://github.com/gtluu/pyBaf2Sql

To get an example Jupyter Notebook file for MSIGen, you can download it from this folder https://github.com/LabLaskin/MSIGen/tree/main/other_files, which also contains other example files and the CLI tool .py file.

Installing Python through Anaconda
-----------------------------------
If you do not have Python installed, we recommend installing the Anaconda distribution of Python, which includes the conda package manager and a large number of scientific packages. You can download Anaconda from https://www.anaconda.com/download. If you want a Guided User Interface (GUI), you will need to make and account with Anaconda then download Anaconda Navigator. If you do not want to make an account and/or are comforable with using a Command Line Interface, you can instead download Miniconda from https://www.anaconda.com/download/success. 

Once it is installed, open Anaconda Prompt and run the following to create a new environment named MSIGen with Python 3.11, activate it, and then install MSIGen into it:

.. code:: bash

   conda create -n MSIGen python=3.11 -y

   conda activate MSIGen

   pip install MSIGen

You will need to activate MSIGen each time you want to use it.

Installing Git
----------------

If need to run any of the commands above that use Git, you can download Git from https://git-scm.com/downloads. If you are using Anaconda Prompt, you will need to restart it after installing Git to be able to use Git commands.

Using Jupyter Notebook
-----------------------
If you want to use MSIGen in a Jupyter notebook, you can install Jupyter notebook using pip as shown above. You can open jupyter notebooks in jupyterlab by running the following command in Anaconda Prompt after activating the MSIGen environment:

.. code:: bash

   jupyter lab

This will open a new tab in your default web browser that shows the Jupyter notebook file browser. You can navigate to the folder where you want to create or open a notebook, then click "New" in the top right and select "Python 3" to create a new notebook. You can also click on an existing notebook file (with a .ipynb extension) to open it.

You can instead use VSCode to run jupyter notebooks with more flexibility. VSCode can be installed from https://code.visualstudio.com/download. After installing it, you will need to install the Python extension from Microsoft by clicking on the Extensions icon on the left sidebar (it looks like four squares with one square separated from the others) and searching "Python" and installing the one made by Microsoft. Next open the folder where you downloaded the MSIGen Jupyter Notebook file (https://github.com/LabLaskin/MSIGen/blob/main/other_files/MSIGen_jupyter.ipynb) clicking File in the top left and then clicking Open Folder.
You may have to select a different Python environment if MSIGen is not loaded by default. You can do this by clicking on the Python version in the top right corner of the window and selecting the MSIGen environment. You may have to click on "Enter interpreter path" and then "Find..." to navigate to the python.exe file in the Scripts folder of your MSIGen environment if it does not show up in the list. You can find the location of your MSIGen environment by running `conda info --envs` in Anaconda Prompt after activating the MSIGen environment. The python.exe file will be in the Scripts folder inside the environment folder.

