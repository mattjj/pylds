from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided
from functools import wraps

from lds_messages import \
    kalman_filter as _kalman_filter, \
    rts_smoother as _rts_smoother, \
    filter_and_sample as _filter_and_sample, \
    E_step as _E_step, \
    kalman_info_filter as _kalman_info_filter


def _argcheck(mu_init, sigma_init, A, sigma_states, C, sigma_obs, data):
    def ensure_3D(X):
        X = np.require(X,dtype=np.float64, requirements='C')
        assert 2 <= X.ndim <= 3
        if X.ndim == 3:
            return X
        else:
            T = data.shape[0]
            return as_strided(X, shape=(T,)+X.shape, strides=(0,)+X.strides)

    A, sigma_states, C, sigma_obs = \
        map(ensure_3D, (A, sigma_states, C, sigma_obs))
    data = np.require(data, dtype=np.float64, requirements='C')

    return mu_init, sigma_init, A, sigma_states, C, sigma_obs, data


def _wrap(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*_argcheck(*args,**kwargs))
    return wrapped

kalman_filter = _wrap(_kalman_filter)
rts_smoother = _wrap(_rts_smoother)
filter_and_sample = _wrap(_filter_and_sample)
E_step = _wrap(_E_step)
kalman_info_filter = _wrap(_kalman_info_filter)

