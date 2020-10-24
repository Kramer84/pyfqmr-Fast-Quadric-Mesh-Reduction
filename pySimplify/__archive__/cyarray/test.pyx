from cython.parallel cimport prange

from cyarray cimport LongVector

cdef LongVector x = LongVector._create(10000)
cdef:
    long i, total

with nogil:
    total = 0
    for i in range(10000):
        LongVector.cappend(x, 10 * i)
    for i in prange(10000):
        x.set(i, x.get(i) / 10)
        total += x.get(i)
print(len(x), total)