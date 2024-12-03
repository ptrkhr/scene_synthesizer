# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# Standard Library
import os
import pathlib
import re
import sys

# Third Party
from sphinx.ext import apidoc

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, root / "src")

# -- Project information -----------------------------------------------------

project = "Scene Synthesizer"
copyright = "2021-2024, NVIDIA"
author = "Clemens Eppner et al."

from scene_synthesizer import __version__ as synth_version
version = synth_version
release = version

# -- Run sphinx-apidoc -------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html docs _build/docs`.
# See Issue: https://github.com/[[rtfd/readthedocs.org/issues/1139

# output_dir = os.path.join(root, "docs", "_api")
# module_dir = os.path.join(root, "src", "scene_synthesizer")

# apidoc_args = [
#     "--implicit-namespaces",
#     "--force",
#     "--separate",
#     "--module-first",
#     "-o",
#     f"{output_dir}",
#     f"{module_dir}",
# ]

# try:
#     apidoc.main(apidoc_args)
#     print("Running `sphinx-apidoc` complete!")
# except Exception as e:
#     print(f"ERROR: Running `sphinx-apidoc` failed!\n{e}")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions that are shipped with Sphinx (named 'sphinx.ext.*') or your
# custom ones.

# See: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
# NOTE: For more extensions, see
# * https://www.sphinx-doc.org/en/master/usage/extensions/index.html
# * https://matplotlib.org/sampledoc/extensions.html
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinxcontrib.bibtex",
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# List of warning types to suppress
suppress_warnings = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'bizstyle'
# html_theme = 'agogo'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]

html_logo = "_static/kitchen_sketch_small.png"

# -- Options for extensions --------------------------------------------------

# sphinx.ext.autodoc options
# --------------------------
autoclass_content = "init"
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_warningiserror = False
suppress_warnings.extend(["autodoc"])

# mathjax options
# ---------------
# NOTE (roflaherty): See
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#confval-mathjax_config
# http://docs.mathjax.org/en/latest/options/index.html#configuring-mathjax
# https://stackoverflow.com/a/60497853
mathjax_config = {"TeX": {"Macros": {}}}

with open("mathsymbols.tex", "r") as f:
    for line in f:
        macros = re.findall(r"\\(DeclareRobustCommand|newcommand){\\(.*?)}(\[(\d)\])?{(.+)}", line)
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax_config["TeX"]["Macros"][macro[1]] = "{" + macro[4] + "}"
            else:
                mathjax_config["TeX"]["Macros"][macro[1]] = ["{" + macro[4] + "}", int(macro[3])]

# sphinx.ext.todo options
# -----------------------
todo_include_todos = True

# sphinx_rtd_theme options
# ------------------------
# html_theme_options = {"navigation_depth": 1}

autodoc_preserve_defaults = True

# sphinxcontrib-bibtex setting
bibtex_bibfiles = ["../paper/paper.bib"]
bibtex_default_style = 'unsrt'