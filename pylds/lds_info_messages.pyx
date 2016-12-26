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


# TODO instead of specializing last step in info filtering and rts, we could
# instead just pad the input J's and h's by zeroes


#################################
#  information-form operations  #
#################################

def kalman_info_filter(
    double[:,:] J_init, double[:] h_init, double log_Z_init,
    double[:,:,:] J_pair_11, double[:,:,:] J_pair_21, double[:,:,:] J_pair_22,
    double[:,:] h_pair_1, double[:,:] h_pair_2, double[:] log_Z_pair,
    double[:,:,:] J_node, double[:,:] h_node, double[:] log_Z_node):

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
    cdef double lognorm = 0

    # Initialize
    lognorm += log_Z_init

    # run filter forwards
    for t in range(T-1):
        lognorm += info_condition_on(
            J_predict, h_predict, J_node[t], h_node[t], log_Z_node[t],
            filtered_Js[t], filtered_hs[t])
        lognorm += info_predict(
            filtered_Js[t], filtered_hs[t],
            J_pair_11[t], J_pair_21[t], J_pair_22[t],
            h_pair_1[t], h_pair_2[t], log_Z_pair[t],
            J_predict, h_predict,
            temp_n, temp_nn, temp_nn2)
    lognorm += info_condition_on(
        J_predict, h_predict, J_node[T-1], h_node[T-1], log_Z_node[T-1],
        filtered_Js[T-1], filtered_hs[T-1])
    lognorm += info_lognorm(
        filtered_Js[T-1], filtered_hs[T-1], temp_n, temp_nn)

    return lognorm, np.asarray(filtered_Js), np.asarray(filtered_hs)

def info_E_step(
    double[:,::1] J_init, double[::1] h_init, double log_Z_init,
    double[:,:,:] J_pair_11, double[:,:,:] J_pair_21, double[:,:,:] J_pair_22,
    double[:,:] h_pair_1, double[:,:] h_pair_2, double[:] log_Z_pair,
    double[:,:,:] J_node, double[:,:] h_node, double[:] log_Z_node):

    # allocate temporaries and internals
    cdef int T = J_node.shape[0], n = J_node.shape[1]
    cdef int t

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
    cdef double[:,:,::1] ExnxT = np.empty((T-1,n,n))  # 'n' for next
    cdef double lognorm = 0.

    # initialize
    lognorm += log_Z_init

    # run filter forwards
    predict_Js[0,:,:] = J_init
    predict_hs[0,:] = h_init
    for t in range(T-1):
        lognorm += info_condition_on(
            predict_Js[t], predict_hs[t], J_node[t], h_node[t], log_Z_node[t],
            filtered_Js[t], filtered_hs[t])
        lognorm += info_predict(
            filtered_Js[t], filtered_hs[t],
            J_pair_11[t], J_pair_21[t], J_pair_22[t],
            h_pair_1[t], h_pair_2[t], log_Z_pair[t],
            predict_Js[t+1], predict_hs[t+1],
            temp_n, temp_nn, temp_nn2)
    lognorm += info_condition_on(
        predict_Js[T-1], predict_hs[T-1], J_node[T-1], h_node[T-1], log_Z_node[T-1],
        filtered_Js[T-1], filtered_hs[T-1])
    lognorm += info_lognorm(
        filtered_Js[T-1], filtered_hs[T-1], temp_n, temp_nn)

    # run info-form rts update backwards
    # overwriting the filtered params with smoothed ones
    info_to_distn(
        filtered_Js[T-1], filtered_hs[T-1],
        smoothed_mus[T-1], smoothed_sigmas[T-1])
    for t in range(T-2,-1,-1):
        info_rts_backward_step(
            J_pair_11[t], J_pair_21[t], J_pair_22[t],
            h_pair_1[t], h_pair_2[t],
            predict_Js[t+1], filtered_Js[t], filtered_Js[t+1],  # filtered_Js[t] is mutated
            predict_hs[t+1], filtered_hs[t], filtered_hs[t+1],  # filtered_hs[t] is mutated
            smoothed_mus[t], smoothed_mus[t+1], smoothed_sigmas[t], ExnxT[t],
            temp_n, temp_nn, temp_nn2)

    return lognorm, np.asarray(smoothed_mus), \
        np.asarray(smoothed_sigmas), np.swapaxes(ExnxT, 1, 2)


def info_sample(
    double[:,::1] J_init, double[::1] h_init, double log_Z_init,
    double[:,:,:] J_pair_11, double[:,:,:] J_pair_21, double[:,:,:] J_pair_22,
    double[:,:] h_pair_1, double[:,:] h_pair_2, double[:] log_Z_pair,
    double[:,:,:] J_node, double[:,:] h_node, double[:] log_Z_node):

    cdef int T = J_node.shape[0], n = J_node.shape[1]
    cdef int t

    cdef double[:,:,::1] filtered_Js = np.empty((T,n,n))
    cdef double[:,::1] filtered_hs = np.empty((T,n))
    cdef double[:,:,::1] predict_Js = np.empty((T,n,n))
    cdef double[:,::1] predict_hs = np.empty((T,n))

    cdef double[::1]   temp_n   = np.empty((n,), order='F')
    cdef double[::1,:] temp_nn  = np.empty((n,n),order='F')
    cdef double[::1,:] temp_nn2 = np.empty((n,n),order='F')

    # allocate output
    cdef double[:,::1] randseq = np.random.randn(T,n)
    cdef double lognorm = 0.

    # dgemv requires these things
    cdef int inc = 1
    cdef double neg1 = -1., one = 1., zero = 0.

    # initialize
    lognorm += log_Z_init

    # run filter forwards
    predict_Js[0,:,:] = J_init
    predict_hs[0,:] = h_init
    for t in range(T-1):
        lognorm += info_condition_on(
            predict_Js[t], predict_hs[t], J_node[t], h_node[t], log_Z_node[t],
            filtered_Js[t], filtered_hs[t])
        lognorm += info_predict(
            filtered_Js[t], filtered_hs[t],
            J_pair_11[t], J_pair_21[t], J_pair_22[t],
            h_pair_1[t], h_pair_2[t], log_Z_pair[t],
            predict_Js[t+1], predict_hs[t+1],
            temp_n, temp_nn, temp_nn2)
    lognorm += info_condition_on(
        predict_Js[T-1], predict_hs[T-1], J_node[T-1], h_node[T-1], log_Z_node[T-1],
        filtered_Js[T-1], filtered_hs[T-1])
    lognorm += info_lognorm(
        filtered_Js[T-1], filtered_hs[T-1], temp_n, temp_nn)

    # sample backward
    info_sample_gaussian(filtered_Js[T-1], filtered_hs[T-1], randseq[T-1])
    for t in range(T-2,-1,-1):
        # temp_n = h_1 - J_12^T x_{t+1}
        # J_pair_21 is C-major, so it is actually J12 to blas!
        dcopy(&n, &h_pair_1[t,0], &inc, &temp_n[0], &inc)
        dgemv('N', &n, &n, &neg1, &J_pair_21[t,0,0], &n, &randseq[t+1,0],
              &inc, &one, &temp_n[0], &inc)

        info_condition_on(
            filtered_Js[t], filtered_hs[t], J_pair_11[t], temp_n, 0,
            filtered_Js[t], filtered_hs[t])
        info_sample_gaussian(filtered_Js[t], filtered_hs[t], randseq[t])

    return lognorm, np.asarray(randseq)


###########################
#  information-form util  #
###########################

cdef inline double info_condition_on(
    double[:,:] J1, double[:] h1,
    double[:,:] J2, double[:] h2,
    double log_Z,
    double[:,:] Jout, double[:] hout,
    ) nogil:
    cdef int n = J1.shape[0]
    cdef int i

    for i in range(n):
        hout[i] = h1[i] + h2[i]

    for i in range(n):
        for j in range(n):
            Jout[i,j] = J1[i,j] + J2[i,j]

    return log_Z


cdef inline double info_predict(
    double[:,:] J, double[:] h,
    double[:,:] J11, double[:,:] J21, double[:,:] J22,
    double[:] h1, double[:] h2, double log_Z,
    double[:,:] Jpredict, double[:] hpredict,
    double[:] temp_n, double[:,:] temp_nn, double[:,:] temp_nn2,
    ) nogil:

    # NOTE: J21 is in C-major order, so BLAS and LAPACK function calls mark it as
    # transposed

    cdef int n = J.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double one = 1., zero = 0., neg1 = -1., lognorm = 0.

    # Copy J to temp_nn and add J_11
    dcopy(&nn, &J[0,0], &inc, &temp_nn[0,0], &inc)
    daxpy(&nn, &one, &J11[0,0], &inc, &temp_nn[0,0], &inc)

    # Copy h to temp_n and add h_1
    dcopy(&n, &h[0], &inc, &temp_n[0], &inc)
    daxpy(&n, &one, &h1[0], &inc, &temp_n[0], &inc)

    # Initialize J_predict to J_22, h_predict to h_2, and temp_nn2 with J_21
    dcopy(&nn, &J22[0,0], &inc, &Jpredict[0,0], &inc)
    dcopy(&n, &h2[0], &inc, &hpredict[0], &inc)
    dcopy(&nn, &J21[0,0], &inc, &temp_nn2[0,0], &inc)

    # Inputs: temp_nn = J_{t|t} + J_11
    #         temp_n  = h_{t|t} + h_1
    # L = cholesky(J_{t|t} + J_11)
    # v = solve_triangular(L, h_{t|t} + h_1)
    # lognorm = 1./2 * np.dot(v,v) - np.sum(np.log(np.diag(L)))
    lognorm += info_lognorm_destructive(temp_nn, temp_n)
    # mutates temp_n and temp_nn
    # Now temp_nn = chol(J+J11), temp_n = chol(J+J11)^{-1} (h+h1)
    # Solve again so that temp_n = (J+J11)^{-1} (h+h1)
    dtrtrs('L', 'T', 'N', &n, &inc, &temp_nn[0,0], &n, &temp_n[0], &n, &info)
    # Finally, subtract J21 (J+J11)^{-1} (h+h1)
    # NOTE: transpose because J21 is in C-major order
    dgemv('T', &n, &n, &neg1, &J21[0,0], &n, &temp_n[0], &inc, &one, &hpredict[0], &inc)

    # Solve again to get temp_nn2 = (J+J11)^{-1}J_12
    dtrtrs('L', 'N', 'N', &n, &n, &temp_nn[0,0], &n, &temp_nn2[0,0], &n, &info)
    # Finally, subtract to get Jp = J22 - J21 (Jf+J11)^{-1} J21
    # TODO this call aliases pointers, should really call dsyrk and copy lower to upper
    dgemm('T', 'N', &n, &n, &n, &neg1, &temp_nn2[0,0], &n, &temp_nn2[0,0], &n, &one, &Jpredict[0,0], &n)
    # dsyrk('L', 'T', &n, &n, &neg1, &temp_nn2[0,0], &n, &one, &Jpredict[0,0], &n)

    return lognorm + log_Z


cdef inline double info_lognorm_destructive(double[:,:] J, double[:] h) nogil:
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


cdef inline double info_lognorm(
    double[:,:] J, double[:] h,
    double[:] temp_n, double[:,:] temp_nn,
    ) nogil:
    cdef int n = J.shape[0]
    cdef int nn = n*n, inc = 1

    dcopy(&nn, &J[0,0], &inc, &temp_nn[0,0], &inc)
    dcopy(&n, &h[0], &inc, &temp_n[0], &inc)

    return info_lognorm_destructive(temp_nn, temp_n)


cdef inline void info_rts_backward_step(
    double[:,:] J11, double[:,:] J21, double[:,:] J22,
    double[:] h1, double[:] h2,
    double[:,:] Jpred_tp1, double[:,:] Jfilt_t, double[:,:] Jsmooth_tp1,  # Jfilt_t is mutated!
    double[:] hpred_tp1, double[:] hfilt_t, double[:] hsmooth_tp1,  # hfilt_t is mutated!
    double[:] mu_t, double[:] mu_tp1, double[:,:] sigma_t, double[:,:] ExnxT,
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

    # temp_nn = Jsmooth_tp1 - J_pred_tp1 + J22
    dcopy(&nn, &Jsmooth_tp1[0,0], &inc, &temp_nn[0,0], &inc)
    daxpy(&nn, &neg1, &Jpred_tp1[0,0], &inc, &temp_nn[0,0], &inc)
    daxpy(&nn, &one, &J22[0,0], &inc, &temp_nn[0,0], &inc)

    # temp_nn2 = J_12.T (recall C order)
    copy_transpose(n, n, &J21[0,0], &temp_nn2[0,0])

    # temp_nn2 = temp_nn^{-1} temp_nn2
    #          = (Jsmooth_tp1 - J_pred_tp1 + J22)^{-1/2} J_12.T
    dpotrf('L', &n, &temp_nn[0,0], &n, &info)
    dtrtrs('L', 'N', 'N', &n, &n, &temp_nn[0,0], &n, &temp_nn2[0,0], &n, &info)

    # Jfilt_t = J_filt_t + J11
    daxpy(&nn, &one, &J11[0,0], &inc, &Jfilt_t[0,0], &inc)

    # J_filt_t = J_filt_t + J11 - J_12 (J_smooth - J_pred_tp1 + J_22)^{-1} J_12.T
    dgemm('T', 'N', &n, &n, &n, &neg1, &temp_nn2[0,0], &n, &temp_nn2[0,0], &n, &one, &Jfilt_t[0,0], &n)


    # hfilt_t = h_filt_t + h1
    daxpy(&n, &one, &h1[0], &inc, &hfilt_t[0], &inc)

    # temp_n = h_smooth_tp1 - h_pred_tp1 + h2
    dcopy(&n, &hsmooth_tp1[0], &inc, &temp_n[0], &inc)
    daxpy(&n, &neg1, &hpred_tp1[0], &inc, &temp_n[0], &inc)
    daxpy(&n, &one, &h2[0], &inc, &temp_n[0], &inc)

    # temp_n = (Jsmooth_tp1 - J_pred_tp1 + J22)^{-1} (h_smooth_tp1 - h_pred_tp1 + h2)
    dpotrs('L', &n, &inc, &temp_nn[0,0], &n, &temp_n[0], &n, &info)

    # h_filt_t = h_filt_t + h1 - J_12.T (Jsmooth_tp1 - J_pred_tp1 + J22)^{-1} (h_smooth_tp1 - h_pred_tp1 + h2)
    dgemv('N', &n, &n, &neg1, &J21[0,0], &n, &temp_n[0], &inc, &one, &hfilt_t[0], &inc)

    # Convert to distribution form
    info_to_distn(Jfilt_t, hfilt_t, mu_t, sigma_t)

    # Compute expected sufficient statistics
    dgemm('T', 'N', &n, &n, &n, &neg1, &J21[0,0], &n, &sigma_t[0,0], &n, &zero, &ExnxT[0,0], &n)
    dpotrs('L', &n, &n, &temp_nn[0,0], &n, &ExnxT[0,0], &n, &info)
    dger(&n, &n, &one, &mu_tp1[0], &inc, &mu_t[0], &inc, &ExnxT[0,0], &n)


cdef inline void info_to_distn(
    double[:,:] J, double[:] h, double[:] mu, double[:,:] Sigma,
    ) nogil:
    cdef int n = J.shape[0]
    cdef int nn = n*n
    cdef int inc = 1, info = 0
    cdef double zero = 0., one = 1.

    dcopy(&nn, &J[0,0], &inc, &Sigma[0,0], &inc)
    dpotrf('L', &n, &Sigma[0,0], &n, &info)
    dpotri('L', &n, &Sigma[0,0], &n, &info)
    copy_upper_lower(n, &Sigma[0,0])  # NOTE: 'L' in Fortran order, but upper for C order
    dgemv('N', &n, &n, &one, &Sigma[0,0], &n, &h[0], &inc, &zero, &mu[0], &inc)


cdef inline void info_sample_gaussian(
    double[:,:] J, double[:] h,
    double[:] randvec,
    ) nogil:
    cdef int n = h.shape[0]
    cdef int inc = 1, info = 0
    cdef double one = 1.

    dpotrf('L', &n, &J[0,0], &n, &info)
    dtrtrs('L', 'T', 'N', &n, &inc, &J[0,0], &n, &randvec[0], &n, &info)
    dpotrs('L', &n, &inc, &J[0,0], &n, &h[0], &n, &info)
    daxpy(&n, &one, &h[0], &inc, &randvec[0], &inc)


###################
#  test bindings  #
###################

def info_predict_test(J,h,J11,J21,J22,h1,h2,logZ, Jpredict,hpredict):
    temp_n = np.random.randn(*h.shape)
    temp_nn = np.random.randn(*J.shape)
    temp_nn2 = np.random.randn(*J.shape)

    return info_predict(J,h,J11,J21,J22,h1,h2,logZ,Jpredict,hpredict,temp_n,temp_nn,temp_nn2)
