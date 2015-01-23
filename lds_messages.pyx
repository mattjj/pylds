# distutils: extra_compile_args = -O2 -w
# distutils: language = c++
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

import numpy as np
cimport numpy as np
cimport cython

from blas_lapack cimport r_gemv, r_gemm, r_symv, r_symm, \
        matrix_qform, chol_update, chol_downdate
from blas_lapack cimport dsymm, dcopy, dgemm, dpotrf, \
        dgemv, dpotrs, daxpy, dtrtrs, dsyrk, dtrmv

# TODO RTS smoother
# TODO use info form when p > n
# TODO E_step (smooth augmented state)
# TODO cholesky update/downdate versions (square root filter)
# TODO write single-precision codepath?
# TODO check that sequence of BLAS/LAPACK calls are compiled to just that and
# there isn't any extra checking or dereferencing going on

# NOTE: I tried the dsymm / dsyrk version and it was slower, even for larger p!
# NOTE: for symmetric matrices, F/C order doesn't matter
# NOTE: clean version of the code is like 1.5-3% slower due to struct passing
# overhead, but much nicer

def kalman_filter(
    double[::1] mu_init, double[:,:] sigma_init,
    double[:,:] A, double[:,:] sigma_states,
    double[:,:] C, double[:,:] sigma_obs,
    double[:,::1] data):

    ### allocate temporaries and internals
    cdef int t, T = data.shape[0]
    cdef int n = C.shape[1], p = C.shape[0]

    cdef double[::1] mu_predict = np.copy(mu_init,order='F')
    cdef double[::1,:] sigma_predict = np.copy(sigma_init,order='F')

    cdef double[::1,:] _A = np.asfortranarray(A)
    cdef double[::1,:] _C = np.asfortranarray(C)

    cdef double[::1,:] temp_pp = np.empty((p,p),order='F')
    cdef double[::1,:] temp_pn = np.empty((p,n),order='F')
    cdef double[::1]   temp_p  = np.empty((p,), order='F')
    cdef double[::1,:] temp_nn = np.empty((n,n),order='F')

    ### allocate output
    cdef double[:,:] filtered_mus = np.empty((T,n))
    cdef double[:,:,:] filtered_sigmas = np.empty((T,n,n))

    ### run filter forwards
    for t in range(T):
        condition_on(
            mu_predict, sigma_predict, _C, sigma_obs, data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            filtered_mus[t], filtered_sigmas[t], _A, sigma_states,
            mu_predict, sigma_predict,
            temp_nn)

    return np.asarray(filtered_mus), np.asarray(filtered_sigmas)

def rts_smoother(
    double[::1] mu_init, double[:,:] sigma_init,
    double[:,:] A, double[:,:] sigma_states,
    double[:,:] C, double[:,:] sigma_obs,
    double[:,::1] data):

    ### allocate temporaries and internals
    cdef int t, T = data.shape[0]
    cdef int n = C.shape[1], p = C.shape[0]

    cdef double[:,:] mu_predicts = np.empty((T+1,n))
    cdef double[:,:,:] sigma_predicts = np.empty((T+1,n,n))

    cdef double[::1,:] _A = np.asfortranarray(A)
    cdef double[::1,:] _C = np.asfortranarray(C)

    cdef double[::1,:] temp_pp  = np.empty((p,p),order='F')
    cdef double[::1,:] temp_pn  = np.empty((p,n),order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),order='F')
    cdef double[::1,:] temp_nn2 = np.empty((n,n),order='F')
    cdef double[::1]   temp_p   = np.empty((p,), order='F')

    ### allocate output
    cdef double[:,:] smoothed_mus = np.empty((T,n))
    cdef double[:,:,:] smoothed_sigmas = np.empty((T,n,n))

    ### run filter forwards, saving predictions
    mu_predicts[0] = mu_init
    sigma_predicts[0] = sigma_init
    for t in range(T):
        condition_on(
            mu_predicts[t], sigma_predicts[t], _C, sigma_obs, data[t],
            smoothed_mus[t], smoothed_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            smoothed_mus[t], smoothed_sigmas[t], _A, sigma_states,
            mu_predicts[t+1], sigma_predicts[t+1],
            temp_nn)

    ### run rts update backwards, using predictions
    for t in range(T-2,-1,-1):
        rts_backward_step(
            _A, sigma_states,
            smoothed_mus[t], smoothed_sigmas[t],
            mu_predicts[t+1], sigma_predicts[t+1],
            smoothed_mus[t+1], smoothed_sigmas[t+1],
            temp_nn, temp_nn2)

    return np.asarray(smoothed_mus), np.asarray(smoothed_sigmas)

def filter_and_sample(
    double[::1] mu_init, double[:,:] sigma_init,
    double[:,:] A, double[:,:] sigma_states,
    double[:,:] C, double[:,:] sigma_obs,
    double[:,::1] data):

    ### allocate temporaries and internals
    cdef int t, T = data.shape[0]
    cdef int n = C.shape[1], p = C.shape[0]

    cdef double[::1] mu_predict = np.copy(mu_init,order='F')
    cdef double[::1,:] sigma_predict = np.copy(sigma_init,order='F')

    cdef double[::1,:] _A = np.asfortranarray(A)
    cdef double[::1,:] _C = np.asfortranarray(C)

    cdef double[::1,:] temp_pp = np.empty((p,p),order='F')
    cdef double[::1,:] temp_pn = np.empty((p,n),order='F')
    cdef double[::1]   temp_p  = np.empty((p,), order='F')
    cdef double[::1,:] temp_nn = np.empty((n,n),order='F')
    cdef double[::1]   temp_n  = np.empty((n,), order='F')

    cdef double[:,:] filtered_mus = np.empty((T,n))
    cdef double[:,:,:] filtered_sigmas = np.empty((T,n,n))

    ### allocate output and generate randomness
    cdef double[:,::1] randseq = np.random.randn(T,n)

    ### run filter forwards
    for t in range(T):
        condition_on(
            mu_predict, sigma_predict, _C, sigma_obs, data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            filtered_mus[t], filtered_sigmas[t], _A, sigma_states,
            mu_predict, sigma_predict,
            temp_nn)

    ### sample backwards
    sample_gaussian(filtered_mus[T-1], filtered_sigmas[T-1], randseq[T-1])
    for t in range(T-2,-1,-1):
        condition_on(
            filtered_mus[t], filtered_sigmas[t], _A, sigma_states, randseq[t+1],
            filtered_mus[t], filtered_sigmas[t],
            temp_n, temp_nn, sigma_predict)
        sample_gaussian(filtered_mus[t], filtered_sigmas[t], randseq[t])

    return np.asarray(randseq)

def E_step(
    double[::1] mu_init, double[:,:] sigma_init,
    double[:,:] A, double[:,:] sigma_states,
    double[:,:] C, double[:,:] sigma_obs,
    double[:,::1] data):
    pass

### util

cdef inline void condition_on(
    # inputs
    double[:] mu_x, double[:,:] sigma_x,
    double[::1,:] A, double[:,:] sigma_obs, double[:] y,
    # outputs
    double[:] mu_cond, double[:,:] sigma_cond,
    # temps
    double[:] temp_p, double[:,:] temp_pn, double[:,:] temp_pp,
    ) nogil:
    cdef int n = A.shape[1], p = A.shape[0]
    cdef int nn = n*n, pp = p*p
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1.

    # NOTE: this routine could actually call predict (a pxn version)

    if y[0] != y[0]:
        dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
    else:
        dgemm('N', 'N', &p, &n, &n, &one, &A[0,0], &p, &sigma_x[0,0], &n, &zero, &temp_pn[0,0], &p)
        # dsymm('R','L', &p, &n, &one, &sigma_x[0,0], &n, &A[0,0], &p, &zero, &temp_pn[0,0], &p)
        dcopy(&pp, &sigma_obs[0,0], &inc, &temp_pp[0,0], &inc)
        dgemm('N', 'T', &p, &p, &n, &one, &temp_pn[0,0], &p, &A[0,0], &p, &one, &temp_pp[0,0], &p)
        dpotrf('L', &p, &temp_pp[0,0], &p, &info)

        dcopy(&p, &y[0], &inc, &temp_p[0], &inc)
        dgemv('N', &p, &n, &neg1, &A[0,0], &p, &mu_x[0], &inc, &one, &temp_p[0], &inc)
        dpotrs('L', &p, &inc, &temp_pp[0,0], &p, &temp_p[0], &p, &info)
        if (&mu_x[0] != &mu_cond[0]):
            dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dgemv('T', &p, &n, &one, &temp_pn[0,0], &p, &temp_p[0], &inc, &one, &mu_cond[0], &inc)

        dtrtrs('L', 'N', 'N', &p, &n, &temp_pp[0,0], &p, &temp_pn[0,0], &p, &info)
        if (&sigma_x[0,0] != &sigma_cond[0,0]):
            dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
        dgemm('T', 'N', &n, &n, &p, &neg1, &temp_pn[0,0], &p, &temp_pn[0,0], &p, &one, &sigma_cond[0,0], &n)
        # dsyrk('L','T', &n, &p, &neg1, &temp_pn[0,0], &p, &one, &sigma_cond[0,0], &n)

cdef inline void predict(
    # inputs
    double[:] mu, double[:,:] sigma,
    double[::1,:] A, double[:,:] sigma_states,
    # outputs
    double[:] mu_predict, double[:,:] sigma_predict,
    # temps
    double[:,:] temp_nn,
    ) nogil:
    cdef int n = mu.shape[0]
    cdef int nn = n*n
    cdef int inc = 1
    cdef double one = 1., zero = 0.

    dgemv('N', &n, &n, &one, &A[0,0], &n, &mu[0], &inc, &zero, &mu_predict[0], &inc)

    dgemm('N', 'N', &n, &n, &n, &one, &A[0,0], &n, &sigma[0,0], &n, &zero, &temp_nn[0,0], &n)
    # dsymm('R','L',&n, &n, &one, &filtered_sigmas[t,0,0], &n, &A[0,0], &n, &zero, &temp_nn[0,0], &n)
    dcopy(&nn, &sigma_states[0,0], &inc, &sigma_predict[0,0], &inc)
    dgemm('N', 'T', &n, &n, &n, &one, &temp_nn[0,0], &n, &A[0,0], &n, &one, &sigma_predict[0,0], &n)

cdef inline void sample_gaussian(
    # inputs (which get mutated)
    double[:] mu, double[:,:] sigma,
    # input/output
    double[:] randvec,
    ) nogil:
    cdef int n = mu.shape[0]
    cdef int inc = 1, info = 0
    cdef double one = 1.

    dpotrf('L', &n, &sigma[0,0], &n, &info)
    dtrmv('L', 'N', 'N', &n, &sigma[0,0], &n, &randvec[0], &inc)
    daxpy(&n, &one, &mu[0], &inc, &randvec[0], &inc)

cdef inline void rts_backward_step(
        double[:,:] A, double[:,:] sigma_states,
        double[:] filtered_mu, double[:,:] filtered_sigma, # inputs/outputs
        double[:] next_predict_mu, double[:,:] next_predict_sigma, # mutated inputs!
        double[:] next_smoothed_mu, double[:,:] next_smoothed_sigma,
        double[:,:] temp_nn, double[:,:] temp_nn2,
        ) nogil:

    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1.

    dgemm('N', 'N', &n, &n, &n, &one, &A[0,0], &n, &filtered_sigma[0,0], &n, &zero, &temp_nn[0,0], &n)
    # TODO: could just call dposv directly instead of dpotrf+dpotrs
    dcopy(&nn, &next_predict_sigma[0,0], &inc, &temp_nn2[0,0], &inc)
    dpotrf('L', &n, &temp_nn2[0,0], &n, &info)
    dpotrs('L', &n, &n, &temp_nn2[0,0], &n, &temp_nn[0,0], &n, &info)

    daxpy(&n, &neg1, &next_smoothed_mu[0], &inc, &next_predict_mu[0], &inc)
    dgemv('T', &n, &n, &neg1, &temp_nn[0,0], &n, &next_predict_mu[0], &inc, &one, &filtered_mu[0], &inc)

    daxpy(&nn, &neg1, &next_smoothed_sigma[0,0], &inc, &next_predict_sigma[0,0], &inc)
    dgemm('N', 'N', &n, &n, &n, &neg1, &next_predict_sigma[0,0], &n, &temp_nn[0,0], &n, &zero, &temp_nn2[0,0], &n)
    dgemm('T', 'N', &n, &n, &n, &one, &temp_nn[0,0], &n, &temp_nn2[0,0], &n, &one, &filtered_sigma[0,0], &n)

