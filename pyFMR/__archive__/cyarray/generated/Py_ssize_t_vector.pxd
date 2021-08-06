


cdef struct Py_ssize_t_vector:
    Py_ssize_t* v
    size_t used
    size_t size

cdef Py_ssize_t_vector* make_Py_ssize_t_vector_with_size(size_t size) nogil
cdef Py_ssize_t_vector* make_Py_ssize_t_vector() nogil
cdef int Py_ssize_t_vector_resize(Py_ssize_t_vector* vec) nogil
cdef int Py_ssize_t_vector_append(Py_ssize_t_vector* vec, Py_ssize_t value) nogil
cdef int Py_ssize_t_vector_reserve(Py_ssize_t_vector* vec, size_t new_size) nogil
cdef void Py_ssize_t_vector_reset(Py_ssize_t_vector* vec) nogil

cdef void free_Py_ssize_t_vector(Py_ssize_t_vector* vec) nogil
cdef void print_Py_ssize_t_vector(Py_ssize_t_vector* vec) nogil

cdef class size_tVector(object):
    cdef __cythonbufferdefaults__ = {'ndim' : 1, 'mode':'c'}

    cdef:
        Py_ssize_t_vector* impl
        int flags

    cdef int allocate_storage(self) nogil
    cdef int allocate_storage_with_size(self, size_t size) nogil

    cdef int free_storage(self) nogil
    cdef bint get_should_free(self) nogil
    cdef void set_should_free(self, bint flag) nogil

    cdef Py_ssize_t* get_data(self) nogil

    @staticmethod
    cdef size_tVector _create(size_t size)

    @staticmethod
    cdef size_tVector wrap(Py_ssize_t_vector* vector)

    cdef Py_ssize_t get(self, size_t i) nogil
    cdef void set(self, size_t i, Py_ssize_t value) nogil
    cdef size_t size(self) nogil
    cdef int cappend(self, Py_ssize_t value) nogil

    cdef size_tVector _slice(self, object slice_spec)

    cpdef size_tVector copy(self)

    cpdef int append(self, object value) except *
    cpdef int extend(self, object values) except *

    cpdef int reserve(self, size_t size) nogil

    cpdef int fill(self, Py_ssize_t value) nogil


    cpdef void qsort(self, bint reverse=?) nogil

    cpdef object _to_python(self, Py_ssize_t value)
    cpdef Py_ssize_t _to_c(self, object value) except *