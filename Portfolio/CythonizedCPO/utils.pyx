# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
import warnings
import cython
from cython.parallel cimport prange


cdef double[:] scalar_divide(double a, double[:] b):
    cdef:
        int n = b.shape[0]
        int i
        double[:] out = np.zeros(n, dtype=np.float64)

    for i in prange(n, nogil=True):
        out[i] = a / b[i]
    return out


cdef double[:] is_positive(double[:] projected_weights):
    cdef:
        int n = projected_weights.shape[0]
        int i
        double[:] out = np.zeros(n, dtype=np.float64)

    for i in prange(n, nogil=True):
        out[i] = 1.0 if projected_weights[i] > 0.0 else 0.0
    return out


cdef double dot(double[:] a, double[:] b):
    cdef int n = a.shape[0]
    cdef int i
    cdef double total = 0

    for i in range(n):
        total += a[i] * b[i]
    return total


cdef double[:] subtract(double[:] a, double[:] b):
    cdef int n = a.shape[0]
    cdef int i
    cdef double[:] out = np.zeros(n, dtype=np.float64)

    for i in prange(n, nogil=True):
        out[i] = a[i] - b[i]
    return out


cdef double[:] add(double[:] a, double[:] b):
    cdef int n = a.shape[0]
    cdef int i
    cdef double[:] out = np.zeros(n, dtype=np.float64)

    for i in prange(n, nogil=True):
        out[i] = a[i] + b[i]
    return out


cdef double[:] scalar_mul(double a, double[:] b):
    cdef int n = b.shape[0]
    cdef int i
    cdef double[:] out = np.zeros(n, dtype=np.float64)

    for i in prange(n, nogil=True):
        out[i] = a * b[i]
    return out


cdef double[:] hamard_prod(double[:] a, double[:] b):
    cdef:
        int n = a.shape[0]
        int i
        double[:] out = np.zeros(n, dtype=np.float64)

    for i in prange(n, nogil=True):
        out[i] = a[i] * b[i]
    return out


cdef double[:] abs(double[:] a):
    cdef:
        int n = a.shape[0]
        int i
        double val
        double[:] out = np.zeros(n, dtype=np.float64)

    for i in prange(n, nogil=True):
        val = a[i]
        out[i] = val if val > 0.0 else -val
    return out
