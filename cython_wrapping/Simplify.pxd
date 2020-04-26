# distutils: language = c++
## Cython wrapper to the C++ Simplify package


from libcpp.string cimport string

cdef extern from "Main.cpp":
	pass


cdef extern from "Simplify.h" namespace "Simplify":
	cdef string obj_string 
	cdef cppclass Simplify :
		Simplify() except +
		Simplify()