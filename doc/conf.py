# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'invert'
copyright = '2022, Lukas Hecker'
author = 'Lukas Hecker'
release = '01.01.2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinxemoji.sphinxemoji', 
    'm2r2', 
    'sphinx.ext.autodoc', 
    'sphinx.ext.coverage', 
    'sphinx.ext.napoleon',
    'sphinx_copybutton'
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']

html_theme_options = {
  "collapse_navigation": True,
  "navigation_depth": 2,
  "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/lukethehecker/invert",
            "icon": "fab fa-github-square",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/invert",
            "icon": "fa-solid fa-box",
        },
    ],
}

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise interprise
    "github_user": "lukethehecker",
    "github_repo": "https://github.com/lukethehecker/invert",
    "github_version": "main",
    "doc_path": "doc",
}

import os
import sys
sys.path.insert(0, os.path.abspath('../'))