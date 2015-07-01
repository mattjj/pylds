from __future__ import division
import numpy as np
from numpy.random import randn, rand, randint

from pylds.lds_messages_interface import filter_and_sample, filter_and_sample_randomwalk


##########
#  util  #
##########

def generate_model(n):
    sigmasq_states = rand(n)
    sigmasq_obs = rand(n)

    mu_init = randn(n)
    sigmasq_init = rand(n)

    return sigmasq_states, sigmasq_obs, mu_init, sigmasq_init


def generate_data(sigmasq_states, sigmasq_obs, mu_init, sigmasq_init, T):
    n = sigmasq_states.shape[0]
    x = np.zeros((T+1,n))
    out = np.zeros((T,n))

    staterandseq = randn(T,n)
    emissionrandseq = randn(T,n)

    x[0] = mu_init + np.sqrt(sigmasq_init)*randn(n)
    for t in xrange(T):
        x[t+1] = x[t] + np.sqrt(sigmasq_states)*staterandseq[t]
        out[t] = x[t] + np.sqrt(sigmasq_obs)*emissionrandseq[t]

    return out


def dense_sample_states(sigmasq_states, sigmasq_obs, mu_init, sigmasq_init, data):
    T, n = data.shape

    # construct corresponding dense model
    A = np.eye(n)
    sigma_states = np.diag(sigmasq_states)
    C = np.eye(n)
    sigma_obs = np.diag(sigmasq_obs)
    sigma_init = np.diag(sigmasq_init)

    return filter_and_sample(
        mu_init, sigma_init, A, sigma_states, C, sigma_obs, data)


#####################
#  testing samples  #
#####################

def check_sample(sigmasq_states, sigmasq_obs, mu_init, sigmasq_init, data):
    rngstate = np.random.get_state()
    dense_ll, dense_sample = dense_sample_states(
        sigmasq_states, sigmasq_obs, mu_init, sigmasq_init, data)
    np.random.set_state(rngstate)
    rw_ll, rw_sample = filter_and_sample_randomwalk(
        mu_init, sigmasq_init, sigmasq_states, sigmasq_obs, data)

    assert np.isclose(dense_ll, rw_ll)
    assert np.allclose(dense_sample, rw_sample)


##################################
#  test against dense functions  #
##################################

def test_filter_and_sample():
    for _ in xrange(5):
        n, T = randint(1,10), randint(10,50)
        model = generate_model(n)
        data = generate_data(*(model + (T,)))
        yield (check_sample,) + model + (data,)
