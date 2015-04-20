from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided
from functools import wraps, partial

################################
#  distribution-form wrappers  #
################################

from lds_messages import \
    kalman_filter as _kalman_filter, \
    rts_smoother as _rts_smoother, \
    filter_and_sample as _filter_and_sample, \
    E_step as _E_step


def _ensure_ndim(X,T,ndim):
    X = np.require(X,dtype=np.float64, requirements='C')
    assert ndim-1 <= X.ndim <= ndim
    if X.ndim == ndim:
        assert X.shape[0] == T
        return X
    else:
        return as_strided(X, shape=(T,)+X.shape, strides=(0,)+X.strides)


def _argcheck(mu_init, sigma_init, A, sigma_states, C, sigma_obs, data):
    T = data.shape[0]
    A, sigma_states, C, sigma_obs = \
        map(partial(_ensure_ndim, T=T, ndim=3),
            [A, sigma_states, C, sigma_obs])
    data = np.require(data, dtype=np.float64, requirements='C')
    return mu_init, sigma_init, A, sigma_states, C, sigma_obs, data


def _wrap(func, check):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*check(*args,**kwargs))
    return wrapped


kalman_filter = _wrap(_kalman_filter,_argcheck)
rts_smoother = _wrap(_rts_smoother,_argcheck)
filter_and_sample = _wrap(_filter_and_sample,_argcheck)
E_step = _wrap(_E_step,_argcheck)


###############################
#  information-form wrappers  #
###############################

# from lds_messages import \
#     info_E_step as _info_E_step


def _info_argcheck(J_init, h_init, J_pair_11, J_pair_12, J_pair_22, J_node, h_node):
    T = h_node.shape[0]
    J_pair_11, J_pair_12, J_pair_22, J_node = \
        map(partial(_ensure_ndim, T=T, ndim=3),
            [J_pair_11, J_pair_12, J_pair_22, J_node])
    h_node = np.require(h_node, dtype=np.float64, requirements='C')
    return J_init, h_init, J_pair_11, J_pair_12, J_pair_22, J_node, h_node


# info_E_step = _wrap(_info_E_step, _info_argcheck)


