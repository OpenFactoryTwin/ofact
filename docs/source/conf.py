# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

path_code = os.path.join(os.path.dirname(__file__), '..', '..')  # '..',
path_code = os.path.abspath(path_code)
print(path_code)
sys.path.insert(0, path_code)


project = 'Open Factory Twin (OFacT)'
copyright = '2024, OpenFactoryTwin'
author = 'OFacT Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autoapi.extension', 'sphinx_rtd_theme', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'recommonmark']

# Configure Napoleon
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Configure AutoAPI
autoapi_dirs = ['../../ofact']  # For some reason, this is relative to the docs/source folder and not
# from the added path reachable
autoapi_ignore = ['*/projects/*', '*tests*', '*test*', '*docs*', '*utilities*']
autoapi_template_dir = "/_templates/autoapi"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/ofact - logo.png"
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
