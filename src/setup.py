from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("simplify_wrapper.pyx", language_level = "3",annotate=True),
    include_dirs=[numpy.get_include()]
)