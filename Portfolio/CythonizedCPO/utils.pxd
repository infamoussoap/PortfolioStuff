# cython: language_level=3

cdef double[:] scalar_divide(double a, double[:] b)

cdef double[:] is_positive(double[:] projected_weights)

cdef double dot(double[:] a, double[:] b)

cdef double[:] subtract(double[:] a, double[:] b)

cdef double[:] add(double[:] a, double[:] b)

cdef double[:] scalar_mul(double a, double[:] b)

cdef double[:] hamard_prod(double[:] a, double[:] b)
