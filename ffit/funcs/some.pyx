import time
import numpy
cimport numpy
cimport cython

ctypedef numpy.float_t DTYPE_t

cdef extern from "math.h":
    cpdef double sin(double x)
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[DTYPE_t, ndim=1] sin_func(numpy.ndarray[DTYPE_t, ndim=1] arr, float amp, float freq, float phase, float offset):
    for k in range(len(arr)):
        arr[k] =amp*sin(freq*arr[k]+phase) + offset
    return arr