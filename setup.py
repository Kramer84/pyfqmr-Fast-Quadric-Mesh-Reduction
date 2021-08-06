from setuptools import setup
from setuptools.extension import Extension
import numpy

#from  https://github.com/AshleySetter/HowToPackageCythonAndCppFuncs


# load the version from the file
with open("VERSION", 'r') as fic:
    version = fic.read()

with open('README.md') as f:
    readme = f.read()

extensions = [
    Extension(
    name="pyFMR.SumArray", # name/path of generated .so file
    sources=["cythonSumPackage/SumArrayCython/SumArray.c"], # cython generated c file
    include_dirs = [numpy.get_include()]), # gives access to numpy funcs inside cython code 
    Extension(
    name="cythonSumPackage.SumArrayCythonCpp", # name/path of generated .so file
    sources=["cythonSumPackage/SumArrayC/SumArrayCythonCpp.cpp"], # cython generated cpp file
    include_dirs = [numpy.get_include()], # gives access to numpy funcs inside cython code 
    language="c++",), # tells python that the language of the extension is c++
]

setup(
    name = "pyFMR",
    cmdclass=cmdclass,
    ext_modules =  ext_modules, #cythonize(["src/simplify_wrapper.pyx", "src/__init__.pyx"], language_level = "3",annotate=True),
    include_dirs=[numpy.get_include()],
    version=version,
    description = "cython wrapper around C++ library for fast triangular mesh reduction",
    author="kramer84",
    url = "https://github.com/Kramer84/pyFMR-Fast-quadric-Mesh-Reduction-",
    license='MIT',
    packages= ['pyFMR'],
    #package_dir={
    #'pyFMR':'src'
    #},
    long_description=readme,
    install_requires=['cython','numpy','trimesh'],
    classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Intended Audience :: Science/Research",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering",
                   "License :: OSI Approved :: MIT License",
                   "Natural Language :: English"]
)



