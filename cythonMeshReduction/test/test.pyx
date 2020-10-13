import numpy as np 
import cython
from cython import int as cy_int
from cython import double as cy_double 
from libc.math cimport sin, cos, pow, abs, sqrt, acos, fmax, fmin
from numpy import int32,float64
from numpy cimport int32_t, float64_t
import trimesh as tr
from libc.stdlib cimport malloc, free

ctypedef struct myTestStruct :
    int a
    int b
    int c 

    def A(int u):
        print('a before',a)
        a+=u
        print('a after',a)


class cdef myCLass :
    cdef myTestStruct *_ptr

    def __cinit__(self, int a):
        self._ptr.a =a 

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            free(self._ptr)
            self._ptr = NULL

    # Extension class properties
    @property
    def a(self):
        return self._ptr.a if self._ptr is not NULL else None

    @property
    def b(self):
        return self._ptr.b if self._ptr is not NULL else None











#############################################################################
#############################################################################
   #class Simplify :




