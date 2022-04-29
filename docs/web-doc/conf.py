#Based on - https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/tree/master/docs

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Demo - https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# Problems with imports? Could try `export PYTHONPATH=$PYTHONPATH:`pwd`` from root project dir...
import os, shutil
import sys
sys.path.insert(0, os.path.abspath('../../src/'))  # Source code dir relative to this file

# -- Get Jupyter Notebooks ---------------------------------------------------
def copy_tree(src, tar):
    """Copies over notebooks into the documentation folder, so get around an issue where nbsphinx
    requires notebooks to be in the same folder as the documentation folder
    """    
    if os.path.exists(tar):
        shutil.rmtree(tar)
    shutil.copytree(src, tar)

copy_tree("../../notebooks", "./_copied_over/notebooks")


# -- Project information -----------------------------------------------------

project = 'Spyglass'
# The full version, including alpha/beta/rc tags
release = '1.0.0'
author = 'Loren Frank'
copyright = '2020 - Present, Loren Frank'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    #'sphinx.autodoc.typehints', # Automatically document param types (less noise in class signature)
    'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    'IPython.sphinxext.ipython_console_highlighting',
    'myst_parser',
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = True  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = False  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
#autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False # Remove namespaces from class/method signatures

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'requirements.txt']

nbsphinx_execute = 'never'

nitpick_ignore = [('py:class', 'type')]

datajoint_definitions = ['Table definition:', 'Table definition', 'definition']

def autodoc_process_docstring(app, what, name, obj, options, lines):       
    if len(lines) > 0 and lines[0].strip() in datajoint_definitions: 
        # TODO - Readdress later
        # Replace  Table Definition" with "DataJoint Attributes" 
        formatted_lines = ['Datajoint Attributes']
        formatted_lines.append('\n')
        
        for line in [line.strip() for line in lines if line != ''][1:]:
            formatted_lines.append(line)
            formatted_lines.append('\n')
        lines.clear()
        lines.extend(formatted_lines)

def html_page_context(app, pagename, templatename, context, doctree):
    # TODO - Readdress later
    # Hack to get issue where Sphinx injected "DataJoint Attributes" when there 
    # is no doc_string. Note, Table Definition" was replaced by "DataJoint Attributes" 
    # in autodoc-process-docstring event
    if context and 'body' in context.keys():
        context['body'] = context['body'].replace(
            '<td><p>Datajoint Attributes</p></td>',
            '<td><p></p></td>'
        )


def setup(app):
    # Entry point to autodoc-skip-member
    app.connect('autodoc-process-docstring', autodoc_process_docstring)

    app.connect('html-page-context', html_page_context)

# #https://github.com/CrossNox/m2r2/issues/4
# github_doc_root = 'https://github.com/rtfd/recommonmark/tree/master/doc/'
# def setup(app):
#     app.add_config_value('recommonmark_config', {
#             'url_resolver': lambda url: github_doc_root + url,
#             'auto_toc_tree_section': 'Contents',
#             }, True)
#     app.add_transform('AutoStructify')

# Exclusions
# To exclude a module, use autodoc_mock_imports. Note this may increase build time, a lot.
# (Also, when installing on readthedocs.org, we omit installing Tensorflow and
# Tensorflow Probability so mock them here instead.)
#autodoc_mock_imports = [
    # 'tensorflow',
    # 'tensorflow_probability',
#]
# To exclude a class, function, method or attribute, use autodoc-skip-member. (Note this can also
# be used in reverse, ie. to re-include a particular member that has been excluded.)
# 'Private' and 'special' members (_ and __) are excluded using the Jinja2 templates; from the main
# doc by the absence of specific autoclass directives (ie. :private-members:), and from summary
# tables by explicit 'if-not' statements. Re-inclusion is effective for the main doc though not for
# the summary tables.
# def autodoc_skip_member_callback(app, what, name, obj, skip, options):
#     # This would exclude the Matern12 class and to_default_float function:
#     exclusions = ('Matern12', 'to_default_float')
#     # This would re-include __call__ methods in main doc, previously excluded by templates:
#     inclusions = ('__call__')
#     if name in exclusions:
#         return True
#     elif name in inclusions:
#         return False
#     else:
#         return skip
# def setup(app):
#     # Entry point to autodoc-skip-member
#     app.connect("autodoc-skip-member", autodoc_skip_member_callback)

# -- Options for HTML output -------------------------------------------------

# Pydata theme
#html_theme = 'sphinx_book_theme'
#html_theme = "pydata_sphinx_theme"
#html_theme = 'karma_sphinx_theme'
html_theme = 'furo'
 
#html_logo = "_static/logo-company.png"
#html_theme_options = { "show_prev_next": False}
#html_css_files = ['pydata-custom.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = './images/lorenLabImage.png'
html_logo = './images/lorenLabImage.png'
html_sourcelink_suffix = ''
html_css_files = [
    'css/datajoint_header.css',
]
