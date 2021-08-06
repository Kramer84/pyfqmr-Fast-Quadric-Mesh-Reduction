#from distutils.core import setup, Extension
from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy

# load the version from the file
with open("VERSION", 'r') as fic:
    version = fic.read()

with open('README.md') as f:
    readme = f.read()

setup(
    ext_modules = cythonize(["src/simplify_wrapper.pyx", "src/__init__.pyx"], language_level = "3",annotate=True),
    include_dirs=[numpy.get_include()],
    name = "pyFMR",
    version=version,
    description = "cython wrapper around C++ library for fast triangular mesh reduction",
    author="kramer84",
    url = "https://github.com/Kramer84/pyFMR-Fast-quadric-Mesh-Reduction-",
    license='MIT',
    packages= find_packages(), #['pyFMR'],
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



