#!/usr/bin/env python

#from distutils.core import setup 
#from distutils.extension import Extension
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


setup(
	name = "SimplifyPy",
	ext_modules =cythonize('*.pyx'),
	)



'''setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("SimplifyPy",
							 sources = "SimplifyPyWrap.pyx",
							 language = "c++",
							 include_dirs = ["./"])]s
	)'''