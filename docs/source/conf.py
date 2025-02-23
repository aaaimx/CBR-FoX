# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add your project’s src folder to sys.path
sys.path.insert(0, os.path.abspath('../../src/cbr_fox'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'prueba'
copyright = '2025, prueba'
author = 'prueba'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',      # Support for Google- and NumPy-style docstrings
    'sphinx.ext.viewcode',      # Add links to the source code
    'sphinx.ext.autosummary',   # Generate summary tables automatically
    'sphinx.ext.coverage',      # Coverage testing for docstrings
    'sphinx.ext.todo',          # Support for TODO comments in the docs
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
