from setuptools import setup
from setuptools.extension import Extension

import numpy as np

# setup file based on:  
# https://github.com/AshleySetter/HowToPackageCythonAndCppFuncs

extensions = [
    Extension(
    name         = "pyfqmr.Simplify",        # name/path of generated .so file
    sources      = ["pyfqmr/Simplify.pyx"],  # cython generated cpp file
    include_dirs = [ np.get_include() ],    # ensure numpy can find headers
    language     = "c++"),                  # tells python that the language of the extension is c++
    ]

setup(
    ext_modules      = extensions,
    )



