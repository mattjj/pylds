# distutils: extra_compile_args = -O3 -w
# distutils: include_dirs = pylds/
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

import scipy.linalg.blas
import scipy.linalg.lapack

cdef extern from "f2pyptr.h":
    void *f2py_pointer(object) except NULL

cdef:
    # BLAS
    dcopy_t *dcopy = <dcopy_t*>f2py_pointer(scipy.linalg.blas.dcopy._cpointer)
    dnrm2_t *dnrm2 = <dnrm2_t*>f2py_pointer(scipy.linalg.blas.dnrm2._cpointer)
    daxpy_t *daxpy = <daxpy_t*>f2py_pointer(scipy.linalg.blas.daxpy._cpointer)
    srotg_t *srotg = <srotg_t*>f2py_pointer(scipy.linalg.blas.srotg._cpointer)
    srot_t *srot = <srot_t*>f2py_pointer(scipy.linalg.blas.srot._cpointer)
    drotg_t *drotg = <drotg_t*>f2py_pointer(scipy.linalg.blas.drotg._cpointer)
    drot_t *drot = <drot_t*>f2py_pointer(scipy.linalg.blas.drot._cpointer)
    ddot_t *ddot = <ddot_t*>f2py_pointer(scipy.linalg.blas.ddot._cpointer)

    dsymv_t *dsymv = <dsymv_t*>f2py_pointer(scipy.linalg.blas.dsymv._cpointer)
    dgemv_t *dgemv = <dgemv_t*>f2py_pointer(scipy.linalg.blas.dgemv._cpointer)
    dger_t *dger = <dger_t*>f2py_pointer(scipy.linalg.blas.dger._cpointer)
    dtrmv_t *dtrmv = <dtrmv_t*>f2py_pointer(scipy.linalg.blas.dtrmv._cpointer)

    dgemm_t *dgemm = <dgemm_t*>f2py_pointer(scipy.linalg.blas.dgemm._cpointer)
    dsymm_t *dsymm = <dsymm_t*>f2py_pointer(scipy.linalg.blas.dsymm._cpointer)
    dsyrk_t *dsyrk = <dsyrk_t*>f2py_pointer(scipy.linalg.blas.dsyrk._cpointer)

    # LAPACK
    dposv_t *dposv = <dposv_t*>f2py_pointer(scipy.linalg.lapack.dposv._cpointer)
    dpotrf_t *dpotrf = <dpotrf_t*>f2py_pointer(scipy.linalg.lapack.dpotrf._cpointer)
    dpotrs_t *dpotrs = <dpotrs_t*>f2py_pointer(scipy.linalg.lapack.dpotrs._cpointer)
    dtrtrs_t *dtrtrs = <dtrtrs_t*>f2py_pointer(scipy.linalg.lapack.dtrtrs._cpointer)

