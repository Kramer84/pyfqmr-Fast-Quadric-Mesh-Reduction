import numpy as np 
import cython
from cython import int as cy_int
from cython import double as cy_double 
from libc.math cimport sin, cos, pow, abs, sqrt, acos, fmax, fmin
from numpy import int32,float64
from numpy cimport int32_t, float64_t
import trimesh as tr
from libc.stdlib cimport malloc, free

cdef struct myTestStruct :
    int a
    int b
    int c 

ctypedef myTestStruct theStruct

cpdef object foo():
    cdef theStruct ts
    ts.a = 0
    ts.b = 0
    ts.c = 0


cdef inline int summer(int a, int b, int c):
    return a*a*b+c



