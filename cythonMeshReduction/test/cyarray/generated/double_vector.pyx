cimport cython

from libc.stdlib cimport malloc, realloc, free
from libc cimport *

from cpython.exc cimport PyErr_BadArgument
from cpython.mem cimport PyObject_Malloc, PyObject_Free

from cpython.sequence cimport (
    PySequence_Size, PySequence_Check, PySequence_Fast,
    PySequence_Fast_GET_ITEM, PySequence_Fast_GET_SIZE)
from cpython.slice cimport PySlice_GetIndicesEx

from cpython.float cimport PyFloat_FromDouble, PyFloat_AsDouble
from libc.math cimport fabs

cdef int compare_value_double(const void* a, const void* b) nogil:
    cdef:
        double av, bv
    av = (<double*>a)[0]
    bv = (<double*>b)[0]
    if av < bv:
        return -1
    elif fabs(av - bv) < 1e-6:
        return 0
    else:
        return 1


cdef int compare_value_double_reverse(const void* a, const void* b) nogil:
    return -compare_value_double(a, b)



cdef extern from * nogil:
    int printf (const char *template, ...)
    void qsort (void *base, unsigned short n, unsigned short w, int (*cmp_func)(void*, void*))


DEF GROWTH_RATE = 2
DEF INITIAL_SIZE = 4


cdef double_vector* make_double_vector_with_size(size_t size) nogil:
    cdef:
        double_vector* vec

    vec = <double_vector*>malloc(sizeof(double_vector))
    vec.v = <double*>malloc(sizeof(double) * size)
    vec.size = size
    vec.used = 0

    return vec


cdef double_vector* make_double_vector() nogil:
    return make_double_vector_with_size(INITIAL_SIZE)


cdef int double_vector_resize(double_vector* vec) nogil:
    cdef:
        size_t new_size
        double* v
    new_size = vec.size * GROWTH_RATE
    v = <double*>realloc(vec.v, sizeof(double) * new_size)
    if v == NULL:
        printf("double_vector_resize returned -1\n")
        return -1
    vec.v = v
    vec.size = new_size
    return 0


cdef int double_vector_append(double_vector* vec, double value) nogil:
    if (vec.used + 1) >= vec.size:
        double_vector_resize(vec)
    vec.v[vec.used] = value
    vec.used += 1
    return 0


cdef void free_double_vector(double_vector* vec) nogil:
    free(vec.v)
    free(vec)


cdef void print_double_vector(double_vector* vec) nogil:
    cdef:
        size_t i
    i = 0
    printf("[")
    while i < vec.used:
        printf("%0.6f", vec.v[i])
        if i != (vec.used - 1):
            printf(", ")
        i += 1
    printf("]\n")


cdef void double_vector_reset(double_vector* vec) nogil:
    vec.used = 0


cdef int double_vector_reserve(double_vector* vec, size_t new_size) nogil:
    cdef:
        double* v
    v = <double*>realloc(vec.v, sizeof(double) * new_size)
    if v == NULL:
        printf("double_vector_resize returned -1\n")
        return -1
    vec.v = v
    vec.size = new_size
    if new_size > vec.used:
        vec.used = new_size
    return 0



cdef char* DoubleVector_buffer_type_code = "d"


@cython.final
@cython.freelist(512)
cdef class DoubleVector(object):

    @staticmethod
    cdef DoubleVector _create(size_t size):
        cdef:
            DoubleVector self
        self = DoubleVector.__new__(DoubleVector)
        self.flags = 0
        self.allocate_storage_with_size(size)
        return self

    @staticmethod
    cdef DoubleVector wrap(double_vector* vector):
        cdef:
            DoubleVector self
        self = DoubleVector.__new__(DoubleVector)
        self.flags = 0
        self.impl = vector
        self.set_should_free(False)
        return self

    def __init__(self, seed=None):
        cdef:
            size_t n
        self.flags = 0
        if seed is not None:
            if not PySequence_Check(seed):
                raise TypeError("Must provide a Sequence-like object")
            n = len(seed)
            self.allocate_storage_with_size(n)
            self.extend(seed)
        else:
            self.allocate_storage()

    cdef int allocate_storage_with_size(self, size_t size) nogil:
        if self.impl != NULL:
            if self.flags & VectorStateEnum.should_free:
                free_double_vector(self.impl)
        self.impl = make_double_vector_with_size(size)
        self.flags |= VectorStateEnum.should_free
        return self.impl == NULL

    cdef int allocate_storage(self) nogil:
        if self.impl != NULL:
            if self.flags & VectorStateEnum.should_free:
                free_double_vector(self.impl)
        self.impl = make_double_vector()
        self.flags |= VectorStateEnum.should_free
        return self.impl == NULL

    cdef int free_storage(self) nogil:
        free_double_vector(self.impl)

    cdef bint get_should_free(self) nogil:
        return self.flags & VectorStateEnum.should_free

    cdef void set_should_free(self, bint flag) nogil:
        self.flags &= VectorStateEnum.should_free * flag

    cdef double* get_data(self) nogil:
        return self.impl.v

    cdef double get(self, size_t i) nogil:
        return self.impl.v[i]

    cdef void set(self, size_t i, double value) nogil:
        self.impl.v[i] = value

    cdef size_t size(self) nogil:
        return self.impl.used

    cdef int cappend(self, double value) nogil:
        return double_vector_append(self.impl, value)

    cpdef int append(self, object value) except *:
        cdef:
            double cvalue
        cvalue = self._to_c(value)
        return self.cappend(cvalue)

    cpdef int extend(self, object values) except *:
        cdef:
            size_t i, n
            object fast_seq
        if not PySequence_Check(values):
            raise TypeError("Must provide a Sequence-like object")

        fast_seq = PySequence_Fast(values, "Must provide a Sequence-like object")
        n = PySequence_Fast_GET_SIZE(values)

        for i in range(n):
            if self.append(<object>PySequence_Fast_GET_ITEM(fast_seq, i)) != 0:
                return 1

    cpdef int reserve(self, size_t size) nogil:
        return double_vector_reserve(self.impl, size)

    cpdef int fill(self, double value) nogil:
        cdef:
            size_t i, n
        n = self.size()
        for i in range(n):
            self.set(i, value)
        return 0

    cpdef DoubleVector copy(self):
        cdef:
            DoubleVector dup
            size_t i, n
        n = self.size()
        dup = DoubleVector._create(n)
        for i in range(n):
            dup.cappend(self.get(i))
        return dup

    cdef DoubleVector _slice(self, object slice_spec):
        cdef:
            DoubleVector dup
            Py_ssize_t length, start, stop, step, slice_length, i
        PySlice_GetIndicesEx(
            slice_spec, self.size(), &start, &stop, &step, &slice_length)
        dup = DoubleVector._create(slice_length)
        i = start
        while i < stop:
            dup.cappend(self.get(i))
            i += step
        return dup

    def __dealloc__(self):
        if self.get_should_free():
            self.free_storage()

    def __len__(self):
        return self.size()

    def __iter__(self):
        cdef:
            size_t i, n
        n = self.size()
        for i in range(n):
            yield self._to_python(self.get(i))

    def __getitem__(self, i):
        cdef:
            Py_ssize_t index
            size_t n
        if isinstance(i, slice):
            return self._slice(i)
        index = i
        n = self.size()
        if index < 0:
            index = n + index
        if index > n or index < 0:
            raise IndexError(index)
        return self._to_python(self.get(index))

    def __setitem__(self, i, value):
        cdef:
            Py_ssize_t index
            size_t n
        if isinstance(i, slice):
            raise TypeError("Does not support slice-assignment yet")
        n = self.size()
        index = i
        if index < 0:
            index = n + index
        if index > self.size() or index < 0:
            raise IndexError(i)
        self.set(index, self._to_c(value))

    def __repr__(self):
        return "{self.__class__.__name__}({members})".format(self=self, members=list(self))


    def __getbuffer__(self, Py_buffer* info, int flags):
        # This implementation of getbuffer is geared towards Cython
        # requirements, and does not yet fulfill the PEP.
        # In particular strided access is always provided regardless
        # of flags
        cdef size_t item_count = self.size()

        info.suboffsets = NULL
        info.buf = <char*>self.impl.v
        info.readonly = 0
        info.ndim = 1
        info.itemsize = sizeof(double)
        info.len = info.itemsize * item_count

        info.shape = <Py_ssize_t*> PyObject_Malloc(sizeof(Py_ssize_t) + 2)
        if not info.shape:
            raise MemoryError()
        info.shape[0] = item_count      # constant regardless of resizing
        info.strides = &info.itemsize

        info.format = DoubleVector_buffer_type_code
        info.obj = self

    def __releasebuffer__(self, Py_buffer* info):
        PyObject_Free(info.shape)


    cpdef void qsort(self, bint reverse=False) nogil:
        if reverse:
            qsort(self.get_data(), self.size(), sizeof(double), compare_value_double_reverse)
        else:
            qsort(self.get_data(), self.size(), sizeof(double), compare_value_double)

    cpdef object _to_python(self, double value):
        return PyFloat_FromDouble(value)

    cpdef double _to_c(self, object value) except *:
        return PyFloat_AsDouble(value)