from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided
from functools import wraps, partial

################################
#  distribution-form wrappers  #
################################

from pylds.lds_messages import \
    kalman_filter as _kalman_filter, \
    rts_smoother as _rts_smoother, \
    filter_and_sample as _filter_and_sample, \
    kalman_filter_diagonal as _kalman_filter_diagonal, \
    filter_and_sample_diagonal as _filter_and_sample_diagonal, \
    filter_and_sample_randomwalk as _filter_and_sample_randomwalk, \
    E_step as _E_step


def _ensure_ndim(X,T,ndim):
    X = np.require(X,dtype=np.float64, requirements='C')
    assert ndim-1 <= X.ndim <= ndim
    if X.ndim == ndim:
        assert X.shape[0] == T
        return X
    else:
        return as_strided(X, shape=(T,)+X.shape, strides=(0,)+X.strides)


def _argcheck(mu_init, sigma_init, A, B, sigma_states, C, D, sigma_obs, inputs, data):
    T = data.shape[0]
    A, B, sigma_states, C, D, sigma_obs = \
        map(partial(_ensure_ndim, T=T, ndim=3),
            [A, B, sigma_states, C, D, sigma_obs])
    # Check that the inputs are C ordered and at least 1d
    inputs = np.require(inputs, dtype=np.float64, requirements='C')

    data = np.require(data, dtype=np.float64, requirements='C')
    return mu_init, sigma_init, A, B, sigma_states, C, D, sigma_obs, inputs, data


def _argcheck_diag_sigma_obs(mu_init, sigma_init, A, B, sigma_states, C, D, sigma_obs, inputs, data):
    T = data.shape[0]
    A, B, sigma_states, C, D, = \
        map(partial(_ensure_ndim, T=T, ndim=3),
            [A, B, sigma_states, C, D])
    sigma_obs = _ensure_ndim(sigma_obs, T=T, ndim=2)
    inputs = np.require(inputs, dtype=np.float64, requirements='C')
    data = np.require(data, dtype=np.float64, requirements='C')
    return mu_init, sigma_init, A, B, sigma_states, C, D, sigma_obs, inputs, data


def _argcheck_randomwalk(mu_init, sigma_init, sigmasq_states, sigmasq_obs, data):
    T = data.shape[0]
    sigmasq_states, sigmasq_obs = \
        map(partial(_ensure_ndim, T=T, ndim=2),
            [sigmasq_states, sigmasq_obs])
    data = np.require(data, dtype=np.float64, requirements='C')
    return mu_init, sigma_init, sigmasq_states, sigmasq_obs, data


def _wrap(func, check):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*check(*args,**kwargs))
    return wrapped


kalman_filter = _wrap(_kalman_filter,_argcheck)
rts_smoother = _wrap(_rts_smoother,_argcheck)
filter_and_sample = _wrap(_filter_and_sample,_argcheck)
E_step = _wrap(_E_step,_argcheck)
kalman_filter_diagonal = _wrap(_kalman_filter_diagonal,_argcheck_diag_sigma_obs)
filter_and_sample_diagonal = _wrap(_filter_and_sample_diagonal,_argcheck_diag_sigma_obs)
filter_and_sample_randomwalk = _wrap(_filter_and_sample_randomwalk,_argcheck_randomwalk)


###############################
#  information-form wrappers  #
###############################

from pylds.lds_info_messages import \
    kalman_info_filter as _kalman_info_filter, \
    info_E_step as _info_E_step, \
    info_sample as _info_sample


def _info_argcheck(J_init, h_init, log_Z_init,
                   J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair,
                   J_node, h_node, log_Z_node):
    T = h_node.shape[0]
    assert np.isscalar(log_Z_init)
    J_pair_11, J_pair_21, J_pair_22, J_node = \
        map(partial(_ensure_ndim, T=T, ndim=3),
            [J_pair_11, J_pair_21, J_pair_22, J_node])
    h_pair_1, h_pair_2 = \
        map(partial(_ensure_ndim, T=T, ndim=2),
            [h_pair_1, h_pair_2])
    log_Z_pair = _ensure_ndim(log_Z_pair, T=T-1, ndim=1)
    log_Z_node = _ensure_ndim(log_Z_node, T=T, ndim=1)

    h_node = np.require(h_node, dtype=np.float64, requirements='C')
    return J_init, h_init, log_Z_init, \
           J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair,\
           J_node, h_node, log_Z_node


kalman_info_filter = _wrap(_kalman_info_filter, _info_argcheck)
info_E_step = _wrap(_info_E_step, _info_argcheck)
info_sample = _wrap(_info_sample, _info_argcheck)
