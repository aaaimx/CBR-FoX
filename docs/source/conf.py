# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# Add your project’s src folder to sys.path
sys.path.insert(0, os.path.abspath('../../src'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Case-Based Reasoning Fox'
copyright = '2025, AAAIMX'
author = 'Pérez Pérez Gerardo Arturo, Valdez Ávila Moisés Fernando, Orozco del Castillo Mauricio Gabriel, Recio García Juan Antonio'
release = '1.0.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',      # Support for Google- and NumPy-style docstrings
    'sphinx.ext.viewcode',      # Add links to the source code
    'sphinx.ext.autosummary',   # Generate summary tables automatically
    'sphinx.ext.coverage',      # Coverage testing for docstrings
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
