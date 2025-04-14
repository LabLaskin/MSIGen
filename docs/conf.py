# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from unittest import mock

# Add the path to your package
sys.path.insert(0, os.path.abspath('..'))

# Mock Windows-only modules to prevent RTD build errors
MOCK_MODULES = ['win32api', 'pywintypes', 'win32com.client']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


project = 'MSIGen'
copyright = '2025, Emerson Hernly'
author = 'Emerson Hernly'
release = '0.2.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google-style docstrings
    "sphinx.ext.viewcode",  # Optional: shows source code links
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Include source files for autodoc
autodoc_mock_imports = MOCK_MODULES