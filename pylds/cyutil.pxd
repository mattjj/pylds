# distutils: extra_compile_args = -O3 -w
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

from cython cimport floating
from libc.math cimport sqrt

from scipy.linalg.cython_blas cimport srot, srotg, drot, drotg

# TODO for higher-rank updates, Householder reflections may be preferrable
cdef inline void chol_update(int n, floating *R, floating *z) nogil:
    cdef int k
    cdef int inc = 1
    cdef floating a, b, c, s
    if floating is double:
        for k in range(n):
            a, b = R[k*n+k], z[k]
            drotg(&a,&b,&c,&s)
            drot(&n,&R[k*n],&inc,&z[0],&inc,&c,&s)
    else:
        for k in range(n):
            a, b = R[k*n+k], z[k]
            srotg(&a,&b,&c,&s)
            srot(&n,&R[k*n],&inc,&z[0],&inc,&c,&s)

cdef inline void chol_downdate(int n, floating *R, floating *z) nogil:
    cdef int k, j
    cdef floating rbar
    for k in range(n):
        rbar = sqrt((R[k*n+k] - z[k])*(R[k*n+k] + z[k]))
        for j in range(k+1,n):
            R[k*n+j] = (R[k*n+k]*R[k*n+j] - z[k]*z[j]) / rbar
            z[j] = (rbar*z[j] - z[k]*R[k*n+j]) / R[k*n+k]
        R[k*n+k] = rbar

cdef inline void copy_transpose(int m, int n, floating *x, floating *y) nogil:
    # NOTE: x is (m,n) and stored in Fortran order
    cdef int i, j
    for i in range(m):
        for j in range(n):
            y[n*i+j] = x[m*j+i]

cdef inline void copy_upper_lower(int n, floating *x) nogil:
    cdef int i, j
    for i in range(n):
        for j in range(i):
            x[n*i+j] = x[n*j+i]
            