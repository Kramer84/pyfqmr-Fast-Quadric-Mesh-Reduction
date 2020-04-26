from setuptools 	import setup 
from Cython.Build 	import cythonise 

setup(ext_modules = cythonise("Simplify.pxd"))
