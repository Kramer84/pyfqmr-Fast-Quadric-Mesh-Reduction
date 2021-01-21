from cpython.string cimport PyString_FromStringAndSize, PyString_AsStringAndSize

# mstr, a struct for holding externally owned string data

cdef mstr mstr_from_str(str pystring):
    cdef:
        mstr result
        Py_ssize_t size
    PyString_AsStringAndSize(pystring, &result.string, &size)
    result.size = size
    return result


cdef str mstr_as_str(mstr mystring):
    return PyString_FromStringAndSize(mystring.string, mystring.size)
