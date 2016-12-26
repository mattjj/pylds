# distutils: extra_compile_args = -O2 -w
# distutils: include_dirs = pylds/
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

import numpy as np
from numpy.lib.stride_tricks import as_strided

cimport numpy as np
cimport cython
from libc.math cimport log, sqrt
from numpy.math cimport INFINITY, PI

from scipy.linalg.cython_blas cimport dsymm, dcopy, dgemm, dgemv, daxpy, dsyrk, \
    dtrmv, dger, dnrm2, ddot
from scipy.linalg.cython_lapack cimport dpotrf, dpotrs, dpotri, dtrtrs
from cyutil cimport copy_transpose, copy_upper_lower

# NOTE: because the matrix operations are done in Fortran order but the code
# expects C ordered arrays as input, the BLAS and LAPACK function calls mark
# input matrices as transposed. temporaries, which don't get sliced, are left in
# Fortran order. for symmetric matrices, F/C order doesn't matter.
# NOTE: I tried the dsymm / dsyrk version and it was slower, even for larger p!
# NOTE: using typed memoryview syntax instead of raw pointers is slightly slower
# due to function call struct passing overhead, but much prettier

# TODO try an Eigen version! faster for small matrices (numerically and in
# function call overhead)
# TODO cholesky update/downdate versions (square root filter)


##################################
#  distribution-form operations  #
##################################

def kalman_filter(
    double[:] mu_init, double[:,:] sigma_init,
    double[:,:,:] A, double[:,:,:] B, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] D, double[:,:,:] sigma_obs,
    double[:,:] inputs, double[:,::1] data):

    # allocate temporaries and internals
    cdef int T = C.shape[0], p = C.shape[1], n = C.shape[2]
    cdef int t

    cdef double[::1] mu_predict = np.copy(mu_init)
    cdef double[:,:] sigma_predict = np.copy(sigma_init)

    cdef double[::1,:] temp_pp = np.empty((p,p),order='F')
    cdef double[::1,:] temp_pn = np.empty((p,n),order='F')
    cdef double[::1]   temp_p  = np.empty((p,), order='F')
    cdef double[::1,:] temp_nn = np.empty((n,n),order='F')

    # allocate output
    cdef double[:,::1] filtered_mus = np.empty((T,n))
    cdef double[:,:,::1] filtered_sigmas = np.empty((T,n,n))
    cdef double ll = 0.

    # run filter forwards
    for t in range(T):
        ll += condition_on(
            mu_predict, sigma_predict,
            C[t], D[t], sigma_obs[t],
            inputs[t], data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            filtered_mus[t], filtered_sigmas[t], inputs[t],
            A[t], B[t], sigma_states[t],
            mu_predict, sigma_predict,
            temp_nn)

    return ll, np.asarray(filtered_mus), np.asarray(filtered_sigmas)


def rts_smoother(
    double[::1] mu_init, double[:,::1] sigma_init,
    double[:,:,:] A, double[:,:,:] B, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] D, double[:,:,:] sigma_obs,
    double[:,:] inputs, double[:,::1] data):

    # allocate temporaries and internals
    cdef int T = C.shape[0], p = C.shape[1], n = C.shape[2]
    cdef int t

    cdef double[:,::1] mu_predicts = np.empty((T+1,n))
    cdef double[:,:,:] sigma_predicts = np.empty((T+1,n,n))

    cdef double[::1,:] temp_pp  = np.empty((p,p),order='F')
    cdef double[::1,:] temp_pn  = np.empty((p,n),order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),order='F')
    cdef double[::1,:] temp_nn2 = np.empty((n,n),order='F')
    cdef double[::1]   temp_p   = np.empty((p,), order='F')

    # allocate output
    cdef double[:,::1] smoothed_mus = np.empty((T,n))
    cdef double[:,:,::1] smoothed_sigmas = np.empty((T,n,n))
    cdef double ll = 0.

    # run filter forwards, saving predictions
    mu_predicts[0] = mu_init
    sigma_predicts[0] = sigma_init
    for t in range(T):
        ll += condition_on(
            mu_predicts[t], sigma_predicts[t],
            C[t], D[t], sigma_obs[t],
            inputs[t], data[t],
            smoothed_mus[t], smoothed_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            smoothed_mus[t], smoothed_sigmas[t], inputs[t],
            A[t], B[t], sigma_states[t],
            mu_predicts[t+1], sigma_predicts[t+1],
            temp_nn)

    # run rts update backwards, using predictions
    for t in range(T-2,-1,-1):
        rts_backward_step(
            A[t], sigma_states[t],
            smoothed_mus[t], smoothed_sigmas[t],
            mu_predicts[t+1], sigma_predicts[t+1],
            smoothed_mus[t+1], smoothed_sigmas[t+1],
            temp_nn, temp_nn2)

    return ll, np.asarray(smoothed_mus), np.asarray(smoothed_sigmas)


def filter_and_sample(
    double[:] mu_init, double[:,:] sigma_init,
    double[:,:,:] A, double[:,:,:] B, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] D, double[:,:,:] sigma_obs,
    double[:,:] inputs, double[:,::1] data):

    # allocate temporaries and internals
    cdef int T = C.shape[0], p = C.shape[1], n = C.shape[2]
    cdef int t

    cdef double[::1] mu_predict = np.copy(mu_init)
    cdef double[:,:] sigma_predict = np.copy(sigma_init)

    cdef double[::1,:] temp_pp = np.empty((p,p),order='F')
    cdef double[::1,:] temp_pn = np.empty((p,n),order='F')
    cdef double[::1]   temp_p  = np.empty((p,), order='F')
    cdef double[::1,:] temp_nn = np.empty((n,n),order='F')
    cdef double[::1]   temp_n  = np.empty((n,), order='F')

    cdef double[:,::1] filtered_mus = np.empty((T,n))
    cdef double[:,:,::1] filtered_sigmas = np.empty((T,n,n))

    # allocate output and generate randomness
    cdef double[:,::1] randseq = np.random.randn(T,n)
    cdef double ll = 0.

    # run filter forwards
    for t in range(T):
        ll += condition_on(
            mu_predict, sigma_predict,
            C[t], D[t], sigma_obs[t],
            inputs[t], data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            filtered_mus[t], filtered_sigmas[t], inputs[t],
            A[t], B[t], sigma_states[t],
            mu_predict, sigma_predict,
            temp_nn)

    # sample backwards
    sample_gaussian(filtered_mus[T-1], filtered_sigmas[T-1], randseq[T-1])
    for t in range(T-2,-1,-1):
        condition_on(
            filtered_mus[t], filtered_sigmas[t],
            A[t], B[t], sigma_states[t],
            inputs[t], randseq[t+1],
            filtered_mus[t], filtered_sigmas[t],
            temp_n, temp_nn, sigma_predict)
        sample_gaussian(filtered_mus[t], filtered_sigmas[t], randseq[t])

    return ll, np.asarray(randseq)


def E_step(
    double[:] mu_init, double[:,:] sigma_init,
    double[:,:,:] A, double[:,:,:] B, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] D, double[:,:,:] sigma_obs,
    double[:,:] inputs, double[:,::1] data):

    # NOTE: this is almost the same as the RTS smoother except
    #   1. we collect statistics along the way, and
    #   2. we use the RTS gain matrix to do it

    # allocate temporaries and internals
    cdef int T = C.shape[0], p = C.shape[1], n = C.shape[2]
    cdef int t

    cdef double[:,:] mu_predicts = np.empty((T+1,n))
    cdef double[:,:,:] sigma_predicts = np.empty((T+1,n,n))

    cdef double[::1,:] temp_pp  = np.empty((p,p),order='F')
    cdef double[::1,:] temp_pn  = np.empty((p,n),order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),order='F')
    cdef double[::1,:] temp_nn2 = np.empty((n,n),order='F')
    cdef double[::1]   temp_p   = np.empty((p,), order='F')

    # allocate output
    cdef double[:,::1] smoothed_mus = np.empty((T,n))
    cdef double[:,:,::1] smoothed_sigmas = np.empty((T,n,n))
    cdef double[:,:,::1] ExnxT = np.empty((T-1,n,n))  # 'n' for next
    cdef double ll = 0.

    # run filter forwards, saving predictions
    mu_predicts[0] = mu_init
    sigma_predicts[0] = sigma_init
    for t in range(T):
        ll += condition_on(
            mu_predicts[t], sigma_predicts[t],
            C[t], D[t], sigma_obs[t],
            inputs[t], data[t],
            smoothed_mus[t], smoothed_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            smoothed_mus[t], smoothed_sigmas[t], inputs[t],
            A[t], B[t], sigma_states[t],
            mu_predicts[t+1], sigma_predicts[t+1],
            temp_nn)

    # run rts update backwards, using predictions and setting E[x_t x_{t+1}^T]
    for t in range(T-2,-1,-1):
        rts_backward_step(
            A[t], sigma_states[t],
            smoothed_mus[t], smoothed_sigmas[t],
            mu_predicts[t+1], sigma_predicts[t+1],
            smoothed_mus[t+1], smoothed_sigmas[t+1],
            temp_nn, temp_nn2)
        set_dynamics_stats(
            smoothed_mus[t], smoothed_mus[t+1], smoothed_sigmas[t+1],
            temp_nn, ExnxT[t])

    return ll, np.asarray(smoothed_mus), np.asarray(smoothed_sigmas), np.asarray(ExnxT)


### diagonal emission distributions (D is diagonal)

def kalman_filter_diagonal(
    double[:] mu_init, double[:,:] sigma_init,
    double[:,:,:] A, double[:,:,:] B, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] D, double[:,:] sigma_obs,
    double[:,:] inputs, double[:,::1] data):

    # allocate temporaries and internals
    cdef int T = C.shape[0], p = C.shape[1], n = C.shape[2]
    cdef int t

    cdef double[::1] mu_predict = np.copy(mu_init)
    cdef double[:,:] sigma_predict = np.copy(sigma_init)

    cdef double[::1,:] temp_pn  = np.empty((p,n),  order='F')
    cdef double[::1,:] temp_pn2 = np.empty((p,n),  order='F')
    cdef double[::1,:] temp_pn3 = np.empty((p,n),  order='F')
    cdef double[::1]   temp_p   = np.empty((p,),   order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),  order='F')
    cdef double[::1,:] temp_pk  = np.empty((p,n+1),order='F')
    cdef double[::1,:] temp_nk  = np.empty((n,n+1),order='F')

    # allocate output
    cdef double[:,::1] filtered_mus = np.empty((T,n))
    cdef double[:,:,::1] filtered_sigmas = np.empty((T,n,n))
    cdef double ll = 0.

    # run filter forwards
    for t in range(T):
        ll += condition_on_diagonal(
            mu_predict, sigma_predict,
            C[t], D[t], sigma_obs[t],
            inputs[t], data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_nn, temp_pn, temp_pn2, temp_pn3, temp_pk, temp_nk)
        predict(
            filtered_mus[t], filtered_sigmas[t], inputs[t],
            A[t], B[t], sigma_states[t],
            mu_predict, sigma_predict,
            temp_nn)

    return ll, np.asarray(filtered_mus), np.asarray(filtered_sigmas)


def filter_and_sample_diagonal(
    double[:] mu_init, double[:,:] sigma_init,
    double[:,:,:] A, double[:,:,:] B, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] D, double[:,:] sigma_obs,
    double[:,:] inputs, double[:,::1] data):

    # allocate temporaries and internals
    cdef int T = C.shape[0], p = C.shape[1], n = C.shape[2]
    cdef int t

    cdef double[::1] mu_predict = np.copy(mu_init)
    cdef double[:,:] sigma_predict = np.copy(sigma_init)

    cdef double[::1,:] temp_pn  = np.empty((p,n),  order='F')
    cdef double[::1,:] temp_pn2 = np.empty((p,n),  order='F')
    cdef double[::1,:] temp_pn3 = np.empty((p,n),  order='F')
    cdef double[::1]   temp_p   = np.empty((p,),   order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),  order='F')
    cdef double[::1]   temp_n   = np.empty((n,),   order='F')
    cdef double[::1,:] temp_pk  = np.empty((p,n+1),order='F')
    cdef double[::1,:] temp_nk  = np.empty((n,n+1),order='F')

    cdef double[:,::1] filtered_mus = np.empty((T,n))
    cdef double[:,:,::1] filtered_sigmas = np.empty((T,n,n))

    # allocate output and generate randomness
    cdef double[:,::1] randseq = np.random.randn(T,n)
    cdef double ll = 0.

    # run filter forwards
    for t in range(T):
        ll += condition_on_diagonal(
            mu_predict, sigma_predict,
            C[t], D[t], sigma_obs[t],
            inputs[t], data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_nn, temp_pn, temp_pn2, temp_pn3, temp_pk, temp_nk)
        predict(
            filtered_mus[t], filtered_sigmas[t], inputs[t],
            A[t], B[t], sigma_states[t],
            mu_predict, sigma_predict,
            temp_nn)

    # sample backwards
    sample_gaussian(filtered_mus[T-1], filtered_sigmas[T-1], randseq[T-1])
    for t in range(T-2,-1,-1):
        condition_on(
            filtered_mus[t], filtered_sigmas[t],
            A[t], B[t], sigma_states[t],
            inputs[t], randseq[t+1],
            filtered_mus[t], filtered_sigmas[t],
            temp_n, temp_nn, sigma_predict)
        sample_gaussian(filtered_mus[t], filtered_sigmas[t], randseq[t])

    return ll, np.asarray(randseq)


### random walk (A = I, B is diagonal, C = I, D is diagonal)

def filter_and_sample_randomwalk(
    double[::1] mu_init, double[::1] sigmasq_init, double[:,:] sigmasq_states,
    double[:,:] sigmasq_obs, double[:,::1] data):
    # TODO: the randomwalk code needs to be updated to handle inputs

    # allocate temporaries and internals
    cdef int T = data.shape[0], n = data.shape[1]
    cdef int t

    cdef double[::1] mu_predict = np.copy(mu_init)
    cdef double[::1] sigmasq_predict = np.copy(sigmasq_init)

    cdef double[:,::1] filtered_mus = np.empty((T,n))
    cdef double[:,::1] filtered_sigmasqs = np.empty((T,n))

    # allocate output and generate randomness
    cdef double[:,::1] randseq = np.random.randn(T,n)
    cdef double ll = 0.

    # run filter forwards
    for t in range(T):
        ll += condition_on_randomwalk(
            n, &mu_predict[0], &sigmasq_predict[0], &sigmasq_obs[t,0], &data[t,0],
            &filtered_mus[t,0], &filtered_sigmasqs[t,0])
        predict_randomwalk(
            n, &filtered_mus[t,0], &filtered_sigmasqs[t,0], &sigmasq_states[t,0],
            &mu_predict[0], &sigmasq_predict[0])

    # sample backwards
    sample_diagonal_gaussian(n, &filtered_mus[T-1,0], &filtered_sigmasqs[T-1,0], &randseq[T-1,0])
    for t in range(T-2,-1,-1):
        condition_on_randomwalk(
            n, &filtered_mus[t,0], &filtered_sigmasqs[t,0], &sigmasq_states[t,0], &randseq[t+1,0],
            &filtered_mus[t,0], &filtered_sigmasqs[t,0])
        sample_diagonal_gaussian(n, &filtered_mus[t,0], &filtered_sigmasqs[t,0], &randseq[t,0])

    return ll, np.asarray(randseq)


############################
#  distribution-form util  #
############################

cdef inline double condition_on(
    # prior predictions
    double[:] mu_x, double[:,:] sigma_x,
    # Observation model
    double[:,:] C, double[:,:] D, double[:,:] sigma_obs,
    # Data
    double[:] u, double[:] y,
    # outputs
    double[:] mu_cond, double[:,:] sigma_cond,
    # temps
    double[:] temp_p, double[:,:] temp_pn, double[:,:] temp_pp,
    ) nogil:
    cdef int p = C.shape[0], n = C.shape[1], d = D.shape[1]
    cdef int nn = n*n, pp = p*p
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1., ll = 0.

    if y[0] != y[0]:  # nan check
        dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
        return 0.
    else:
        # NOTE: the C and D arguments are treated as transposed because C and D are
        # assumed to be in C order (row-major order)

        # Compute temp_pn = C * sigma_x
        # and     temp_pp = chol(sigma_obs + C * sigma_x * C.T) = chol(S)
        dgemm('T', 'N', &p, &n, &n, &one, &C[0,0], &n, &sigma_x[0,0], &n, &zero, &temp_pn[0,0], &p)
        # Now temp_pp = sigma_obs
        dcopy(&pp, &sigma_obs[0,0], &inc, &temp_pp[0,0], &inc)
        # temp_pp += temp_pn * C = sigma_obs + C * sigma_x * C.T
        # call this S, as in (18.38) of Murphy
        dgemm('N', 'N', &p, &p, &n, &one, &temp_pn[0,0], &p, &C[0,0], &n, &one, &temp_pp[0,0], &p)
        # temp_pp = cholesky(S, lower=True) = L
        dpotrf('L', &p, &temp_pp[0,0], &p, &info)

        # Compute the residual -- this is where the inputs come in
        dcopy(&p, &y[0], &inc, &temp_p[0], &inc)
        # temp_p -= - C * mu_x = y - C * mu_x
        dgemv('T', &n, &p, &neg1, &C[0,0], &n, &mu_x[0], &inc, &one, &temp_p[0], &inc)
        # temp_p -= - D * u = y - C * mu_x - D * u
        if d > 0:
            dgemv('T', &d, &p, &neg1, &D[0,0], &d, &u[0], &inc, &one, &temp_p[0], &inc)
        # Solve temp_p = temp_pp^{-1} temp_p
        #              = L^{-1} (y - C * mu_x - D * u)
        dtrtrs('L', 'N', 'N', &p, &inc, &temp_pp[0,0], &p, &temp_p[0], &p, &info)
        # log likelihood = -1/2 * ||L^{-1} (y - C * mu_x - D * u)||*2
        ll = (-1./2) * dnrm2(&p, &temp_p[0], &inc)**2

        # Second solve with cholesky
        # temp_p = L.T^{-1} temp_p
        #        = S^{-1} (y - C * mu_x - D * u)
        dtrtrs('L', 'T', 'N', &p, &inc, &temp_pp[0,0], &p, &temp_p[0], &p, &info)

        # Compute the conditional mean
        # mu_cond = mu_x + temp_pn * temp_p
        #         = mu_x + sigma_x * C.T * S^{-1} (y - C * mu_x - D * u)
        # Compare this to (18.31) of Murphy
        if (&mu_x[0] != &mu_cond[0]):
            dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dgemv('T', &p, &n, &one, &temp_pn[0,0], &p, &temp_p[0], &inc, &one, &mu_cond[0], &inc)

        # Compute the conditional covariance
        # sigma_cond = sigma_x - C * sigma_x.T * L.T^{-1} * L^{-1} C.T * sigma_x
        #            = sigma_x - sigma_x * C.T * S^{-1} * C * sigma_x
        # Compare this to (18.32) of Murphy
        #
        # First, temp_pn = temp_pp^{-1} temp_pn
        #                = L^{-1} C.T * sigma_x
        # Then we square this and subtract from sigma_x
        dtrtrs('L', 'N', 'N', &p, &n, &temp_pp[0,0], &p, &temp_pn[0,0], &p, &info)
        if (&sigma_x[0,0] != &sigma_cond[0,0]):
            dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
        # TODO this call aliases pointers, should really call dsyrk and copy lower to upper
        dgemm('T', 'N', &n, &n, &p, &neg1, &temp_pn[0,0], &p, &temp_pn[0,0], &p, &one, &sigma_cond[0,0], &n)

        # Compute log determinant of the covariance by summing log diagonal of cholesky
        ll -= p/2. * log(2.*PI)
        for i in range(p):
            ll -= log(temp_pp[i,i])

        return ll


cdef inline void predict(
    # inputs
    double[:] mu, double[:,:] sigma, double[:] u,
    double[:,:] A, double[:,:] B, double[:,:] sigma_states,
    # outputs
    double[:] mu_predict, double[:,:] sigma_predict,
    # temps
    double[:,:] temp_nn,
    ) nogil:
    cdef int n = mu.shape[0], d = B.shape[1]
    cdef int nn = n*n
    cdef int inc = 1
    cdef double one = 1., zero = 0.

    # NOTE: the A and B arguments are treated as transposed because A and B are assumed to be
    # in C order (row-major order)

    # mu_predict = A * mu
    dgemv('T', &n, &n, &one, &A[0,0], &n, &mu[0], &inc, &zero, &mu_predict[0], &inc)
    # mu_predict += B * u
    if d > 0:
        dgemv('T', &d, &n, &one, &B[0,0], &d, &u[0], &inc, &one, &mu_predict[0], &inc)

    # temp_nn = A * sigma
    dgemm('T', 'N', &n, &n, &n, &one, &A[0,0], &n, &sigma[0,0], &n, &zero, &temp_nn[0,0], &n)
    dcopy(&nn, &sigma_states[0,0], &inc, &sigma_predict[0,0], &inc)
    # sigma_pred = sigma_states + A * sigma * A.T
    dgemm('N', 'N', &n, &n, &n, &one, &temp_nn[0,0], &n, &A[0,0], &n, &one, &sigma_predict[0,0], &n)


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
    double[:] filtered_mu, double[:,:] filtered_sigma,  # inputs/outputs
    double[:] next_predict_mu, double[:,:] next_predict_sigma,  # mutated inputs!
    double[:] next_smoothed_mu, double[:,:] next_smoothed_sigma,
    double[:,:] temp_nn, double[:,:] temp_nn2,  # temps
    ) nogil:
    # filtered_mu and filtered_sigma are m_{t|t} and P_{t|t}, respectively
    # Recall, m_{t|t} = A * m_{t|t-1} + B * u_{t-1},
    #    and, P_{t|t} = A * P_{t|t-1} * A.T + Q_t
    # next_predict_mu and next_predict_sigma are m_{t+1|t} and P_{t+1|t}
    # next_smoothed_mu and next_smoothed_sigma are m_{t+1|T} and P_{t+1|T}

    # NOTE: on exit, temp_nn holds the RTS gain, called G_k.T in the notation of
    # Thm 8.2 of Sarkka 2013 "Bayesian Filtering and Smoothing"

    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1.

    # NOTE: the A argument is treated as transposed because A is assumd to be in C order
    # temp_nn = A * P_{t|t}
    dgemm('T', 'N', &n, &n, &n, &one, &A[0,0], &n, &filtered_sigma[0,0], &n, &zero, &temp_nn[0,0], &n)
    # TODO: could just call dposv directly instead of dpotrf+dpotrs
    # temp_nn2 = P_{t+1|t}
    dcopy(&nn, &next_predict_sigma[0,0], &inc, &temp_nn2[0,0], &inc)
    # temp_nn2 = chol(P_{t|t-1}, lower=True)
    dpotrf('L', &n, &temp_nn2[0,0], &n, &info)
    # temp_nn = temp_nn2^{-1} temp_nn
    #         = (P_{t+1|t})^{-1} A * P_k
    #         = G_t^T
    dpotrs('L', &n, &n, &temp_nn2[0,0], &n, &temp_nn[0,0], &n, &info)

    # next_predict_mu = m_{t+1|t} - m_{t+1|T} (negated version of notes)
    daxpy(&n, &neg1, &next_smoothed_mu[0], &inc, &next_predict_mu[0], &inc)
    # filtered_mu = filtered_mu - temp_nn * next_predict_mu
    #             = m_{t|t} - G_t^T * (m_{t|t-1} - m_{t+1|T})
    #             = m_{t|t} + G_t^T * (m_{t+1|T} - m_{t|t-1})
    #             = m_{t|T}
    dgemv('T', &n, &n, &neg1, &temp_nn[0,0], &n, &next_predict_mu[0], &inc, &one, &filtered_mu[0], &inc)

    # next_predict_sigma = next_predict_sigma - next_smoothed_sigma
    #                    = P_{t+1|t} - P_{t+1|T}
    daxpy(&nn, &neg1, &next_smoothed_sigma[0,0], &inc, &next_predict_sigma[0,0], &inc)
    # temp_nn2 = -next_predict_sigma * temp_nn
    #          = (P_{t+1|T} - P_{t+1|t}) * G_t^T
    dgemm('N', 'N', &n, &n, &n, &neg1, &next_predict_sigma[0,0], &n, &temp_nn[0,0], &n, &zero, &temp_nn2[0,0], &n)
    # filtered_sigma = filtered_sigma + temp_nn * temp_nn2
    #                = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t^T
    #                = P_{t|T}
    dgemm('T', 'N', &n, &n, &n, &one, &temp_nn[0,0], &n, &temp_nn2[0,0], &n, &one, &filtered_sigma[0,0], &n)


cdef inline void set_dynamics_stats(
    double[::1] mk, double[::1] mkn, double[:,::1] Pkns,
    double[::1,:] GkT,
    double[:,::1] ExnxT,
    ) nogil:
    # mk = m_{t|T}
    # mkn = m_{t+1|T}
    # Pkns = P_{t+1|T}
    # GkT is the transpose of the RTS gain G_t = P_t A_t^T P_{t+1|t}^{-1}
    # ExnxT = E[x_{t+1} x_t^T]
    cdef int n = mk.shape[0], inc = 1
    cdef double one = 1., zero = 0.
    # E_xnxT = GkT.T * Pkns
    #        = G_t * P_{t+1|T}
    #        = P_t A_t
    # Compare to Sarkka notes, this is the cross covariance, Cov(xn, x)
    dgemm('T', 'N', &n, &n, &n, &one, &GkT[0,0], &n, &Pkns[0,0], &n, &zero, &ExnxT[0,0], &n)
    # Recall,
    # Cov(xn, x) = E[(x_{t+1} - m_{t+1}) (x_t - m_t)^T]
    #            = E[x_{t+1} x_t^T] - E[m_{t+1} x_t^T] - E[x_{t+1} m_t^T] + m_{t+1} m_t^T
    #            = E[x_{t+1} x_t^T] - m_{t+1} m_t^T
    # Add outer product of means to get E[x_{t+1] x_t^T]
    dger(&n, &n, &one, &mk[0], &inc, &mkn[0], &inc, &ExnxT[0,0], &n)


### diagonal emission distributions

cdef inline double condition_on_diagonal(
    double[:] mu_x, double[:,:] sigma_x,
    double[:,:] C, double[:,:] D, double[:] sigma_obs,
    double[:] u, double[:] y,
    double[:] mu_cond, double[:,:] sigma_cond,
    double[::1] temp_p, double[::1,:] temp_nn,
    double[::1,:] temp_pn, double[::1,:] temp_pn2, double[::1,:] temp_pn3,
    double[::1,:] temp_pk, double[::1,:] temp_nk,
    ) nogil:

    # see Boyd and Vandenberghe, Convex Optimization, Appendix C.4.3 (p. 679)
    # and also https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    # and https://en.wikipedia.org/wiki/Matrix_determinant_lemma

    # an extra temp (temp_pn3) and an extra copy_transpose are needed because C
    # is not stored in Fortran order as solve_diagonal_plus_lowrank requires

    cdef int p = C.shape[0], n = C.shape[1], d = D.shape[1]
    cdef int nn = n*n, pn = p*n
    cdef int inc = 1, info = 0, i
    cdef double one = 1., zero = 0., neg1 = -1., ll = 0.

    if y[0] != y[0]:  # nan check
        dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
        return 0.
    else:
        # NOTE: the C arguments are treated as transposed because C is
        # assumed to be in C order

        # Compute residual y - C * mu_x - D * u
        dcopy(&p, &y[0], &inc, &temp_p[0], &inc)
        dgemv('T', &n, &p, &neg1, &C[0,0], &n, &mu_x[0], &inc, &one, &temp_p[0], &inc)
        if d > 0:
            dgemv('T', &d, &p, &neg1, &D[0,0], &d, &u[0], &inc, &one, &temp_p[0], &inc)

        # Compute conditional mean and variance using low rank plus diagonal code
        dgemm('T', 'N', &p, &n, &n, &one, &C[0,0], &n, &sigma_x[0,0], &n, &zero, &temp_pn[0,0], &p)
        copy_transpose(n, p, &C[0,0], &temp_pn3[0,0])
        dcopy(&p, &temp_p[0], &inc, &temp_pk[0,0], &inc)
        dcopy(&pn, &temp_pn[0,0], &inc, &temp_pk[0,1], &inc)

        ll = -1./2 * solve_diagonal_plus_lowrank(
            sigma_obs, temp_pn3, sigma_x, temp_pk, False,
            temp_nn, temp_pn2, temp_nk)

        if (&mu_x[0] != &mu_cond[0]):
            dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dgemv('T', &p, &n, &one, &temp_pn[0,0], &p, &temp_pk[0,0], &inc, &one, &mu_cond[0], &inc)

        if (&sigma_x[0,0] != &sigma_cond[0,0]):
            dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
        dgemm('T', 'N', &n, &n, &p, &neg1, &temp_pn[0,0], &p, &temp_pk[0,1], &p, &one, &sigma_cond[0,0], &n)

        ll -= 1./2 * ddot(&p, &temp_p[0], &inc, &temp_pk[0,0], &inc)
        ll -= p/2. * log(2*PI)
        return ll


cdef inline double solve_diagonal_plus_lowrank(
    double[:] a, double[:,:] B, double[:,:] C, double[:,:] b, bint C_is_identity,
    double[:,:] temp_nn, double[:,:] temp_pn, double[:,:] temp_nk,
    ) nogil:
    cdef int p = B.shape[0], n = B.shape[1], k = b.shape[1]
    cdef int nn = n*n, inc = 1, info = 0, i, j
    cdef double one = 1., zero = 0., neg1 = -1., logdet = 0.

    # NOTE: on exit, temp_nn is guaranteed to hold chol(C^{-1} + B' A^{-1} B)
    # NOTE: assumes Fortran order for everything

    for j in range(k):
        for i in range(p):
            b[i,j] /= a[i]

    for j in range(n):
        for i in range(p):
            temp_pn[i,j] = B[i,j] / a[i]

    dcopy(&nn, &C[0,0], &inc, &temp_nn[0,0], &inc)
    if not C_is_identity:
        dpotrf('L', &n, &temp_nn[0,0], &n, &info)
        for i in range(n):
            logdet += 2.*log(temp_nn[i,i])
        dpotri('L', &n, &temp_nn[0,0], &n, &info)
    dgemm('T', 'N', &n, &n, &p, &one, &B[0,0], &p, &temp_pn[0,0], &p, &one, &temp_nn[0,0], &n)
    dpotrf('L', &n, &temp_nn[0,0], &n, &info)

    dgemm('T', 'N', &n, &k, &p, &one, &B[0,0], &p, &b[0,0], &p, &zero, &temp_nk[0,0], &n)
    dpotrs('L', &n, &k, &temp_nn[0,0], &n, &temp_nk[0,0], &n, &info)

    dgemm('N', 'N', &p, &k, &n, &neg1, &temp_pn[0,0], &p, &temp_nk[0,0], &n, &one, &b[0,0], &p)

    for i in range(n):
        logdet += 2.*log(temp_nn[i,i])
    for i in range(p):
        logdet += log(a[i])

    return logdet


### identity dynamics and emission distributions (A = I, C = I)

# NOTE: we have to use raw pointers here because numpy (and hence cython's typed
# memoryview checks) doesn't count arrays with a zero stride as possibly being
# C-contiguous

cdef inline double condition_on_randomwalk(
    int n,
    double *mu_x, double *sigmasq_x,
    double *sigmasq_obs, double *y,
    double *mu_cond, double *sigmasq_cond,
    ) nogil:

    # TODO: the randomwalk code needs to be updated to handle inputs

    cdef double ll = -n/2. * log(2.*PI), sigmasq_yi
    cdef int i
    for i in range(n):
        sigmasq_yi = sigmasq_x[i] + sigmasq_obs[i]
        ll -= 1./2 * log(sigmasq_yi)
        ll -= 1./2 * (y[i] - mu_x[i])**2 / sigmasq_yi
        mu_cond[i] = mu_x[i] + sigmasq_x[i] / sigmasq_yi * (y[i] - mu_x[i])
        sigmasq_cond[i] = sigmasq_x[i] - sigmasq_x[i]**2 / sigmasq_yi

    return ll


cdef inline void predict_randomwalk(
    int n,
    double *mu, double *sigmasq, double *sigmasq_states,
    double *mu_predict, double *sigmasq_predict,
    ) nogil:

    cdef int i
    for i in range(n):
        mu_predict[i] = mu[i]
        sigmasq_predict[i] = sigmasq[i] + sigmasq_states[i]


cdef inline void sample_diagonal_gaussian(
    int n,
    double *mu, double  *sigmasq, double *randvec,
    ) nogil:

    cdef int i
    for i in range(n):
        randvec[i] = mu[i] + sqrt(sigmasq[i]) * randvec[i]


###################
#  test bindings  #
###################

def test_condition_on_diagonal(
    double[:] mu_x, double[:,:] sigma_x,
    double[:,:] C, double[:,:] D, double[:] sigma_obs,
    double[:] u, double[:] y,
    double[:] mu_cond, double[:,:] sigma_cond):
    p = y.shape[0]
    n = mu_x.shape[0]
    k = n+1
    temp_p   = np.asfortranarray(np.random.randn(p))
    temp_nn  = np.asfortranarray(np.random.randn(n,n))
    temp_pn  = np.asfortranarray(np.random.randn(p,n))
    temp_pn2 = np.asfortranarray(np.random.randn(p,n))
    temp_pn3 = np.asfortranarray(np.random.randn(p,n))
    temp_pk  = np.asfortranarray(np.random.randn(p,k))
    temp_nk  = np.asfortranarray(np.random.randn(n,k))
    return condition_on_diagonal(
        mu_x, sigma_x, C, D, sigma_obs, u, y, mu_cond, sigma_cond,
        temp_p, temp_nn, temp_pn, temp_pn2, temp_pn3, temp_pk, temp_nk)


def test_solve_diagonal_plus_lowrank(
    double[:] a, double[::1,:] B, double[:,:] C, bint C_is_identity,
    double[::1,:] b):
    p = B.shape[0]
    n = B.shape[1]
    k = b.shape[1]
    temp_nn = np.asfortranarray(np.random.randn(n,n))
    temp_pn = np.asfortranarray(np.random.randn(p,n))
    temp_nk = np.asfortranarray(np.random.randn(n,k))
    return solve_diagonal_plus_lowrank(a,B,C,b,C_is_identity,temp_nn,temp_pn,temp_nk)
