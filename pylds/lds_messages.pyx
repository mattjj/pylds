# distutils: extra_compile_args = -O2 -w
# distutils: include_dirs = pylds/
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

import numpy as np
from numpy.lib.stride_tricks import as_strided

cimport numpy as np
cimport cython
from libc.math cimport log
from numpy.math cimport INFINITY, PI

from blas_lapack cimport dsymm, dcopy, dgemm, dpotrf, \
        dgemv, dpotrs, daxpy, dtrtrs, dsyrk, dtrmv, \
        dger, dnrm2, dpotri, copy_transpose, copy_upper_lower

# NOTE: because the matrix operations are done in Fortran order but the code
# expects C ordered arrays as input, the BLAS and LAPACK function calls mark
# matrices as transposed. temporaries, which don't get sliced, are left in
# Fortran order, and their memoryview types are consistent. for symmetric
# matrices, F/C order doesn't matter.
# NOTE: I tried the dsymm / dsyrk version and it was slower, even for larger p!
# NOTE: using typed memoryview syntax instead of raw pointers is like 1.5-3%
# slower due to struct passing overhead, but much prettier
# NOTE: scipy doesn't expose a dtrsm binding

# TODO cholesky update/downdate versions (square root filter)
# TODO test info smoother/Estep
# TODO factor out filter routines (for parallelism and deduplication)


##################################
#  distribution-form operations  #
##################################

def kalman_filter(
    double[:] mu_init, double[:,:] sigma_init,
    double[:,:,:] A, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] sigma_obs,
    double[:,::1] data):

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
            mu_predict, sigma_predict, C[t], sigma_obs[t], data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            filtered_mus[t], filtered_sigmas[t], A[t], sigma_states[t],
            mu_predict, sigma_predict,
            temp_nn)

    return ll, np.asarray(filtered_mus), np.asarray(filtered_sigmas)


def rts_smoother(
    double[::1] mu_init, double[:,::1] sigma_init,
    double[:,:,:] A, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] sigma_obs,
    double[:,::1] data):

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
            mu_predicts[t], sigma_predicts[t], C[t], sigma_obs[t], data[t],
            smoothed_mus[t], smoothed_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            smoothed_mus[t], smoothed_sigmas[t], A[t], sigma_states[t],
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
    double[:,:,:] A, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] sigma_obs,
    double[:,::1] data):

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
            mu_predict, sigma_predict, C[t], sigma_obs[t], data[t],
            filtered_mus[t], filtered_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            filtered_mus[t], filtered_sigmas[t], A[t], sigma_states[t],
            mu_predict, sigma_predict,
            temp_nn)

    # sample backwards
    sample_gaussian(filtered_mus[T-1], filtered_sigmas[T-1], randseq[T-1])
    for t in range(T-2,-1,-1):
        condition_on(
            filtered_mus[t], filtered_sigmas[t], A[t], sigma_states[t], randseq[t+1],
            filtered_mus[t], filtered_sigmas[t],
            temp_n, temp_nn, sigma_predict)
        sample_gaussian(filtered_mus[t], filtered_sigmas[t], randseq[t])

    return ll, np.asarray(randseq)


def E_step(
    double[:] mu_init, double[:,:] sigma_init,
    double[:,:,:] A, double[:,:,:] sigma_states,
    double[:,:,:] C, double[:,:,:] sigma_obs,
    double[:,::1] data):

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
            mu_predicts[t], sigma_predicts[t], C[t], sigma_obs[t], data[t],
            smoothed_mus[t], smoothed_sigmas[t],
            temp_p, temp_pn, temp_pp)
        predict(
            smoothed_mus[t], smoothed_sigmas[t], A[t], sigma_states[t],
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


############################
#  distribution-form util  #
############################

cdef inline double condition_on(
    # inputs
    double[:] mu_x, double[:,:] sigma_x,
    double[:,:] C, double[:,:] sigma_obs, double[:] y,
    # outputs
    double[:] mu_cond, double[:,:] sigma_cond,
    # temps
    double[:] temp_p, double[:,:] temp_pn, double[:,:] temp_pp,
    ) nogil:
    cdef int p = C.shape[0], n = C.shape[1]
    cdef int nn = n*n, pp = p*p
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1., ll = 0.

    if y[0] != y[0]:  # nan check
        dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
        return 0.
    else:
        # NOTE: the C arguments are treated as transposed because C is
        # assumed to be in C order (row-major order)
        dgemm('T', 'N', &p, &n, &n, &one, &C[0,0], &n, &sigma_x[0,0], &n, &zero, &temp_pn[0,0], &p)
        dcopy(&pp, &sigma_obs[0,0], &inc, &temp_pp[0,0], &inc)
        dgemm('N', 'N', &p, &p, &n, &one, &temp_pn[0,0], &p, &C[0,0], &n, &one, &temp_pp[0,0], &p)
        dpotrf('L', &p, &temp_pp[0,0], &p, &info)

        dcopy(&p, &y[0], &inc, &temp_p[0], &inc)
        dgemv('T', &n, &p, &neg1, &C[0,0], &n, &mu_x[0], &inc, &one, &temp_p[0], &inc)
        dtrtrs('L', 'N', 'N', &p, &inc, &temp_pp[0,0], &p, &temp_p[0], &p, &info)
        ll = (-1./2) * dnrm2(&p, &temp_p[0], &inc)**2
        dtrtrs('L', 'T', 'N', &p, &inc, &temp_pp[0,0], &p, &temp_p[0], &p, &info)
        if (&mu_x[0] != &mu_cond[0]):
            dcopy(&n, &mu_x[0], &inc, &mu_cond[0], &inc)
        dgemv('T', &p, &n, &one, &temp_pn[0,0], &p, &temp_p[0], &inc, &one, &mu_cond[0], &inc)

        dtrtrs('L', 'N', 'N', &p, &n, &temp_pp[0,0], &p, &temp_pn[0,0], &p, &info)
        if (&sigma_x[0,0] != &sigma_cond[0,0]):
            dcopy(&nn, &sigma_x[0,0], &inc, &sigma_cond[0,0], &inc)
        # TODO this call aliases pointers, should really call dsyrk and copy lower to upper
        dgemm('T', 'N', &n, &n, &p, &neg1, &temp_pn[0,0], &p, &temp_pn[0,0], &p, &one, &sigma_cond[0,0], &n)

        ll -= p/2. * log(2.*PI)
        for i in range(p):
            ll -= log(temp_pp[i,i])

        return ll


cdef inline void predict(
    # inputs
    double[:] mu, double[:,:] sigma,
    double[:,:] A, double[:,:] sigma_states,
    # outputs
    double[:] mu_predict, double[:,:] sigma_predict,
    # temps
    double[:,:] temp_nn,
    ) nogil:
    cdef int n = mu.shape[0]
    cdef int nn = n*n
    cdef int inc = 1
    cdef double one = 1., zero = 0.

    # NOTE: the A arguments are treated as transposed because A is assumed to be
    # in C order

    dgemv('T', &n, &n, &one, &A[0,0], &n, &mu[0], &inc, &zero, &mu_predict[0], &inc)

    dgemm('T', 'N', &n, &n, &n, &one, &A[0,0], &n, &sigma[0,0], &n, &zero, &temp_nn[0,0], &n)
    dcopy(&nn, &sigma_states[0,0], &inc, &sigma_predict[0,0], &inc)
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

    # NOTE: on exit, temp_nn holds the RTS gain, called G_k' in the notation of
    # Thm 8.2 of Sarkka 2013 "Bayesian Filtering and Smoothing"

    cdef int n = A.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1.

    # NOTE: the A argument is treated as transposed because A is assumd to be in C order
    dgemm('T', 'N', &n, &n, &n, &one, &A[0,0], &n, &filtered_sigma[0,0], &n, &zero, &temp_nn[0,0], &n)
    # TODO: could just call dposv directly instead of dpotrf+dpotrs
    dcopy(&nn, &next_predict_sigma[0,0], &inc, &temp_nn2[0,0], &inc)
    dpotrf('L', &n, &temp_nn2[0,0], &n, &info)
    dpotrs('L', &n, &n, &temp_nn2[0,0], &n, &temp_nn[0,0], &n, &info)

    daxpy(&n, &neg1, &next_smoothed_mu[0], &inc, &next_predict_mu[0], &inc)
    dgemv('T', &n, &n, &neg1, &temp_nn[0,0], &n, &next_predict_mu[0], &inc, &one, &filtered_mu[0], &inc)

    daxpy(&nn, &neg1, &next_smoothed_sigma[0,0], &inc, &next_predict_sigma[0,0], &inc)
    dgemm('N', 'N', &n, &n, &n, &neg1, &next_predict_sigma[0,0], &n, &temp_nn[0,0], &n, &zero, &temp_nn2[0,0], &n)
    dgemm('T', 'N', &n, &n, &n, &one, &temp_nn[0,0], &n, &temp_nn2[0,0], &n, &one, &filtered_sigma[0,0], &n)


cdef inline void set_dynamics_stats(
    double[::1] mk, double[::1] mkn, double[:,::1] Pkns,
    double[::1,:] GkT,
    double[:,::1] ExnxT,
    ) nogil:

    cdef int n = mk.shape[0], inc = 1
    cdef double one = 1., zero = 0.
    dgemm('T', 'N', &n, &n, &n, &one, &GkT[0,0], &n, &Pkns[0,0], &n, &zero, &ExnxT[0,0], &n)
    dger(&n, &n, &one, &mk[0], &inc, &mkn[0], &inc, &ExnxT[0,0], &n)



#################################
#  information-form operations  #
#################################

def kalman_info_filter(
    double[:,:] J_init, double[:] h_init,
    double[:,:,:] J_pair_11, double[:,:,:] J_pair_21, double[:,:,:] J_pair_22,
    double[:,:,:] J_node, double[:,:] h_node):

    # NOTE: returned lognorm does not include base measure terms

    # allocate temporaries and internals
    cdef int T = J_node.shape[0], n = J_node.shape[1]
    cdef int t

    cdef double[:,:] J_predict = np.copy(J_init)
    cdef double[:] h_predict = np.copy(h_init)

    cdef double[::1]   temp_n   = np.empty((n,), order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),order='F')
    cdef double[::1,:] temp_nn2 = np.empty((n,n),order='F')

    # allocate output
    cdef double[:,:,::1] filtered_Js = np.empty((T,n,n))
    cdef double[:,::1] filtered_hs = np.empty((T,n))
    cdef double lognorm = 0.

    # run filter forwards
    for t in range(T-1):
        info_condition_on(
            J_predict, h_predict, J_node[t], h_node[t],
            filtered_Js[t], filtered_hs[t])
        lognorm += info_predict(
            filtered_Js[t], filtered_hs[t], J_pair_11[t], J_pair_21[t], J_pair_22[t],
            J_predict, h_predict,
            temp_n, temp_nn, temp_nn2)
    info_condition_on(
        J_predict, h_predict, J_node[T-1], h_node[T-1],
        filtered_Js[T-1], filtered_hs[T-1])
    lognorm += info_lognorm_copy(
        filtered_Js[T-1], filtered_hs[T-1], temp_n, temp_nn)

    return lognorm, np.asarray(filtered_Js), np.asarray(filtered_hs)


def info_E_step(
    double[:,:] J_init, double[:] h_init,
    double[:,:,:] J_pair_11, double[:,:,:] J_pair_21, double[:,:,:] J_pair_22,
    double[:,:,:] J_node, double[:,:] h_node):
    # NOTE: uses two-filter strategy

    # allocate temporaries and internals
    cdef int T = J_node.shape[0], n = J_node.shape[1]
    cdef int t

    cdef double[:,:] J_predict = np.copy(J_init)
    cdef double[:] h_predict = np.copy(h_init)

    cdef double[:,:,::1] filtered_Js = np.empty((T,n,n))
    cdef double[:,::1] filtered_hs = np.empty((T,n))
    cdef double[:,:,::1] predict_Js = np.empty((T,n,n))
    cdef double[:,::1] predict_hs = np.empty((T,n))

    cdef double[::1]   temp_n   = np.empty((n,), order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),order='F')
    cdef double[::1,:] temp_nn2 = np.empty((n,n),order='F')

    # allocate output
    cdef double[:,::1] smoothed_mus = np.empty((T,n))
    cdef double[:,:,::1] smoothed_sigmas = np.empty((T,n,n))
    cdef double[:,:,::1] Cov_xnxs = np.empty((T-1,n,n))  # 'n' for next
    cdef double lognorm = 0.

    # run filter forwards
    predict_Js[0] = np.copy(J_init)
    predict_hs[0] = np.copy(h_init)
    for t in range(T-1):
        info_condition_on(
            predict_Js[t], predict_hs[t], J_node[t], h_node[t],
            filtered_Js[t], filtered_hs[t])
        lognorm += info_predict(
            filtered_Js[t], filtered_hs[t], J_pair_11[t], J_pair_21[t], J_pair_22[t],
            predict_Js[t+1], predict_hs[t+1],
            temp_n, temp_nn, temp_nn2)
    info_condition_on(
        J_predict, h_predict, J_node[T-1], h_node[T-1],
        filtered_Js[T-1], filtered_hs[T-1])
    lognorm -= info_lognorm_copy(
        filtered_Js[T-1], filtered_hs[T-1], temp_n, temp_nn)
    lognorm += n/2. * log(2*PI)

    # run info-form rts update backwards
    # overwriting the filtered params with smoothed ones
    for t in range(T-2,-1,-1):
        info_rts_backward_step(
            J_pair_11[t], J_pair_21[t], J_pair_22[t],
            predict_Js[t+1], filtered_Js[t], filtered_Js[t+1],  # filtered_Js[t] is mutated
            predict_hs[t+1], filtered_hs[t], filtered_hs[t+1],  # filtered_hs[t] is mutated
            smoothed_mus[t], smoothed_sigmas[t], Cov_xnxs[t],
            temp_n, temp_nn, temp_nn2)

    return lognorm, np.asarray(smoothed_mus), np.asarray(smoothed_sigmas), np.asarray(Cov_xnxs)


###########################
#  information-form util  #
###########################

cdef inline void info_condition_on(
    double[:,:] J1, double[:] h1,
    double[:,:] J2, double[:] h2,
    double[:,:] Jout, double[:] hout,
    ) nogil:
    cdef int n = J1.shape[0]
    cdef int i

    for i in range(n):
        hout[i] = h1[i] + h2[i]

    for i in range(n):
        for j in range(n):
            Jout[i,j] = J1[i,j] + J2[i,j]


cdef inline double info_lognorm(double[:,:] J, double[:] h) nogil:
    # NOTE: mutates input to chol(J) and solve_triangular(chol(J),h), resp.

    cdef int n = J.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double lognorm = 0.

    dpotrf('L', &n, &J[0,0], &n, &info)
    dtrtrs('L', 'N', 'N', &n, &inc, &J[0,0], &n, &h[0], &n, &info)

    lognorm += (1./2) * dnrm2(&n, &h[0], &inc)**2
    for i in range(n):
        lognorm -= log(J[i,i])
    lognorm += n/2. * log(2*PI)

    return lognorm


cdef inline double info_lognorm_copy(
    double[:,:] J, double[:] h,
    double[:] temp_n, double[:,:] temp_nn,
    ) nogil:
    cdef int n = J.shape[0]
    cdef int nn = n*n, inc = 1

    dcopy(&nn, &J[0,0], &inc, &temp_nn[0,0], &inc)
    dcopy(&n, &h[0], &inc, &temp_n[0], &inc)

    return info_lognorm(temp_nn, temp_n)


cdef inline double info_predict(
    double[:,:] J, double[:] h, double[:,:] J11, double[:,:] J21, double[:,:] J22,
    double[:,:] Jpredict, double[:] hpredict,
    double[:] temp_n, double[:,:] temp_nn, double[:,:] temp_nn2,
    ) nogil:

    # NOTE: J21 is in C-major order, so BLAS and LAPACK function calls mark it as
    # transposed

    cdef int n = J.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1., lognorm = 0.

    dcopy(&nn, &J[0,0], &inc, &temp_nn[0,0], &inc)
    daxpy(&nn, &one, &J11[0,0], &inc, &temp_nn[0,0], &inc)
    dcopy(&nn, &J22[0,0], &inc, &Jpredict[0,0], &inc)
    dcopy(&nn, &J21[0,0], &inc, &temp_nn2[0,0], &inc)
    dcopy(&n, &h[0], &inc, &temp_n[0], &inc)

    lognorm += info_lognorm(temp_nn, temp_n)  # mutates temp_n and temp_nn
    dtrtrs('L', 'T', 'N', &n, &inc, &temp_nn[0,0], &n, &temp_n[0], &n, &info)
    # NOTE: transpose because J21 is in C-major order
    dgemv('T', &n, &n, &neg1, &J21[0,0], &n, &temp_n[0], &inc, &zero, &hpredict[0], &inc)

    dtrtrs('L', 'N', 'N', &n, &n, &temp_nn[0,0], &n, &temp_nn2[0,0], &n, &info)
    # TODO this call aliases pointers, should really call dsyrk and copy lower to upper
    dgemm('T', 'N', &n, &n, &n, &neg1, &temp_nn2[0,0], &n, &temp_nn2[0,0], &n, &one, &Jpredict[0,0], &n)
    # dsyrk('L', 'T', &n, &n, &neg1, &temp_nn2[0,0], &n, &one, &Jpredict[0,0], &n)

    return lognorm


cdef inline void info_rts_backward_step(
    double[:,:] J11, double[:,:] J21, double[:,:] J22,
    double[:,:] Jpred_tp1, double[:,:] Jfilt_t, double[:,:] Jsmooth_tp1,  # Jfilt_t is mutated!
    double[:] hpred_tp1, double[:] hfilt_t, double[:] hsmooth_tp1,  # hfilt_t is mutated!
    double[:] mu_t, double[:,:] sigma_t, double[:,:] Cov_xnx,
    double[:] temp_n, double[:,:] temp_nn, double[:,:] temp_nn2,
    ) nogil:

    # NOTE: this function mutates Jfilt_t and hfilt_t to be Jsmooth_t and
    # hsmooth_t, respectively
    # NOTE: J21 is in C-major order, so BLAS and LAPACK function calls mark it as
    # transposed

    cdef int n = J11.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1.

    dcopy(&nn, &Jsmooth_tp1[0,0], &inc, &temp_nn[0,0], &inc)
    daxpy(&nn, &neg1, &Jpred_tp1[0,0], &inc, &temp_nn[0,0], &inc)
    daxpy(&nn, &one, &J22[0,0], &inc, &temp_nn[0,0], &inc)
    dpotrf('L', &n, &temp_nn[0,0], &n, &info)

    copy_transpose(J21, temp_nn2)
    dtrtrs('L', 'N', 'N', &n, &n, &temp_nn[0,0], &n, &temp_nn2[0,0], &n, &info)
    daxpy(&nn, &one, &J11[0,0], &inc, &Jfilt_t[0,0], &inc)
    dgemm('T', 'N', &n, &n, &n, &neg1, &temp_nn2[0,0], &n, &temp_nn2[0,0], &n, &one, &Jfilt_t[0,0], &n)

    dcopy(&n, &hsmooth_tp1[0], &inc, &temp_n[0], &inc)
    daxpy(&n, &neg1, &hpred_tp1[0], &inc, &temp_n[0], &inc)
    dpotrs('L', &n, &inc, &temp_nn[0,0], &n, &temp_n[0], &n, &info)
    dgemv('N', &n, &n, &neg1, &J21[0,0], &n, &temp_n[0], &inc, &one, &hfilt_t[0], &inc)

    dcopy(&nn, &Jfilt_t[0,0], &inc, &sigma_t[0,0], &inc)
    dpotrf('L', &n, &sigma_t[0,0], &n, &info)
    dcopy(&n, &hfilt_t[0], &inc, &mu_t[0], &inc)
    dpotrs('L', &n, &inc, &sigma_t[0,0], &n, &mu_t[0], &n, &info)
    dpotri('L', &n, &sigma_t[0,0], &n, &info)
    copy_upper_lower(sigma_t)

    dgemm('T', 'N', &n, &n, &n, &neg1, &J21[0,0], &n, &sigma_t[0,0], &n, &zero, &Cov_xnx[0,0], &n)
    dpotrs('L', &n, &n, &temp_nn[0,0], &n, &Cov_xnx[0,0], &n, &info)


### testing

def info_predict_test(J,h,J11,J21,J22,Jpredict,hpredict):
    temp_n = np.random.randn(*h.shape)
    temp_nn = np.random.randn(*J.shape)
    temp_nn2 = np.random.randn(*J.shape)

    return info_predict(J,h,J11,J21,J22,Jpredict,hpredict,temp_n,temp_nn,temp_nn2)


def info_rts_test(
        J11, J21, J22, Jpred_tp1, Jfilt_t, Jsmooth_tp1, hpred_tp1, hfilt_t,
        hsmooth_tp1, mu_t, sigma_t, Cov_xnx):
    temp_n = np.zeros_like(mu_t)
    temp_nn = np.zeros_like(sigma_t)
    temp_nn2 = np.zeros_like(sigma_t)

    return info_rts_backward_step(
          J11, J21, J22, Jpred_tp1, Jfilt_t, Jsmooth_tp1,
          hpred_tp1, hfilt_t, hsmooth_tp1, mu_t, sigma_t, Cov_xnx,
          temp_n, temp_nn, temp_nn2)

