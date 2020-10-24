
cdef enum VectorStateEnum:
    should_free = 1

include "generated/double_vector.pxd"
include "generated/long_vector.pxd"
include "generated/Py_ssize_t_vector.pxd"
