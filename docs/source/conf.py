# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # Adjust the path as needed

sys.path.insert(0, os.path.abspath('../..'))

project = 'CBR-FoX'
copyright = '2024, Perez Perez Gerardo Arturo, Valdez Avila Moises Fernando'
author = 'Perez Perez Gerardo Arturo, Valdez Avila Moises Fernando'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supports NumPy-style docstrings
    'sphinx.ext.autosummary',  # Generates summary tables for modules
    'sphinx.ext.viewcode'  # Adds links to source code
]
autodoc_mock_imports = ["sktime", "numpy"]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,  # Even if no docstring is present
    'private-members': True,  # Include functions starting with _
    'special-members': '__init__',  # Show constructors
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
