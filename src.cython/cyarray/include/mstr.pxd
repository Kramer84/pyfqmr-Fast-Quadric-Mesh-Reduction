
# mstr, a struct for holding externally owned string data

cdef struct mstr:
    char* string
    size_t size