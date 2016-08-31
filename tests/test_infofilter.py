from __future__ import division
import numpy as np
from numpy.random import randn, randint

from pylds.lds_messages_interface import kalman_filter, kalman_info_filter, \
    E_step, info_E_step
from pylds.lds_info_messages import info_predict_test
from pylds.states import LDSStates


##########
#  util  #
##########

def blockarray(*args,**kwargs):
    return np.array(np.bmat(*args,**kwargs),copy=False)


def cumsum(l, strict=False):
    if not strict:
        return np.cumsum(l)
    else:
        return np.cumsum(l) - l[0]


def rand_psd(n,k=None):
    k = k if k else n
    out = randn(n,k)
    return np.atleast_2d(out.dot(out.T))


def blockdiag(mats):
    assert all(m.shape[0] == m.shape[1] for m in mats)
    ns = [m.shape[0] for m in mats]
    starts, stops, n = cumsum(ns,strict=True), cumsum(ns), sum(ns)
    out = np.zeros((n,n))
    for start, stop, mat in zip(starts, stops, mats):
        out[start:stop,start:stop] = mat
    return out


def info_to_mean(J,h):
    mu, Sigma = np.linalg.solve(J,h), np.linalg.inv(J)
    return mu, Sigma


def mean_to_info(mu,Sigma):
    J = np.linalg.inv(Sigma)
    h = np.linalg.solve(Sigma,mu)
    return J, h


def spectral_radius(A):
    return max(np.abs(np.linalg.eigvals(A)))


def generate_model(n, p, d):
    A = randn(n,n)
    A /= 2.*spectral_radius(A)  # ensures stability
    assert spectral_radius(A) < 1.
    B = randn(n,d)
    sigma_states = np.random.randn(n,n)
    sigma_states = np.dot(sigma_states, sigma_states.T)

    C = randn(p,n)
    D = randn(p,d)
    sigma_obs = np.random.randn(p,p)
    sigma_obs = np.dot(sigma_obs, sigma_obs.T)

    mu_init = randn(n)
    sigma_init = rand_psd(n)

    return A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init


def generate_data(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, T):
    p, n, d = C.shape[0], C.shape[1], B.shape[1]
    x = np.zeros((T+1,n))
    u = np.random.randn(T,d)
    out = np.zeros((T,p))

    Ldyn = np.linalg.cholesky(sigma_states)
    Lobs = np.linalg.cholesky(sigma_obs)

    staterandseq = randn(T,n)
    emissionrandseq = randn(T,p)

    x[0] = np.random.multivariate_normal(mu_init,sigma_init)
    for t in range(T):
        x[t+1] = A.dot(x[t]) + B.dot(u[t]) + Ldyn.dot(staterandseq[t])
        out[t] = C.dot(x[t]) + D.dot(u[t]) + Lobs.dot(emissionrandseq[t])

    return out, u


def info_params(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, data, inputs):
    J_init = np.linalg.inv(sigma_init)
    h_init = np.linalg.solve(sigma_init, mu_init)

    J_pair_11 = A.T.dot(np.linalg.solve(sigma_states,A))
    J_pair_21 = -np.linalg.solve(sigma_states,A)
    J_pair_22 = np.linalg.inv(sigma_states)

    h_pair_1 = -inputs.dot(B.T).dot(np.linalg.solve(sigma_states,A))
    h_pair_2 = inputs.dot(np.linalg.solve(sigma_states, B).T)

    J_node = C.T.dot(np.linalg.solve(sigma_obs,C))
    h_node = np.einsum('ik,ij,tj->tk',
                       C, np.linalg.inv(sigma_obs),
                       (data - inputs.dot(D.T)))

    return J_init, h_init, J_pair_11, J_pair_21, J_pair_22, \
        h_pair_1, h_pair_2, J_node, h_node


def dense_infoparams(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, data, inputs):
    p, n = C.shape
    T = data.shape[0]

    J_init, h_init, J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, J_node, h_node = \
        info_params(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, data, inputs)

    h = h_node
    h[0] += h_init
    h[:-1] += h_pair_1[:-1]
    h[1:] += h_pair_2[:-1]
    h = h.ravel()

    J = np.kron(np.eye(T), J_node)
    pairblock = blockarray([[J_pair_11, J_pair_21.T], [J_pair_21, J_pair_22]])
    for t in range(0,n*(T-1),n):
        J[t:t+2*n,t:t+2*n] += pairblock
    J[:n, :n] += J_init

    assert J.shape == (T*n, T*n)
    assert h.shape == (T*n,)

    return J, h


##########################
#  testing info_predict  #
##########################

def py_info_predict(J,h,J11,J21,J22,h1,h2):
    Jnew = J + J11
    Jpredict = J22 - J21.dot(np.linalg.solve(Jnew,J21.T))
    hnew = h + h1
    hpredict = h2-J21.dot(np.linalg.solve(Jnew,hnew))
    lognorm = -1./2*np.linalg.slogdet(Jnew)[1] + 1./2*hnew.dot(np.linalg.solve(Jnew,hnew)) \
        + J.shape[0]/2.*np.log(2*np.pi)
    return Jpredict, hpredict, lognorm


def py_info_predict2(J,h,J11,J21,J22,h1,h2):
    n = J.shape[0]
    bigJ = blockarray([[J11, J21.T], [J21, J22]]) + blockdiag([J, np.zeros_like(J)])
    bigh = np.concatenate([h+h1,h2])
    mu, Sigma = info_to_mean(bigJ, bigh)
    Jpredict = np.linalg.inv(Sigma[n:,n:])
    hpredict = np.linalg.solve(Sigma[n:,n:],mu[n:])
    return Jpredict, hpredict


def cy_info_predict(J,h,J11,J21,J22,h1,h2):
    Jpredict = np.zeros_like(J)
    hpredict = np.zeros_like(h)

    lognorm = info_predict_test(J,h,J11,J21,J22,h1,h2,Jpredict,hpredict)

    return Jpredict, hpredict, lognorm


def check_info_predict(J,h,J11,J21,J22,h1,h2):
    py_Jpredict, py_hpredict, py_lognorm = py_info_predict(J,h,J11,J21,J22,h1,h2)
    cy_Jpredict, cy_hpredict, cy_lognorm = cy_info_predict(J,h,J11,J21,J22,h1,h2)

    assert np.allclose(py_Jpredict, cy_Jpredict)
    assert np.allclose(py_hpredict, cy_hpredict)
    assert np.allclose(py_lognorm, cy_lognorm)

    py2_Jpredict, py2_hpredict = py_info_predict2(J,h,J11,J21,J22,h1,h2)
    assert np.allclose(py2_Jpredict, cy_Jpredict)
    assert np.allclose(py2_hpredict, cy_hpredict)


def test_info_predict():
    for _ in range(5):
        n = randint(1,20)
        J = rand_psd(n)
        h = randn(n)

        bigJ = rand_psd(2*n)
        J11, J21, J22 = map(np.copy,[bigJ[:n,:n], bigJ[n:,:n], bigJ[n:,n:]])

        h1 = randn(n)
        h2 = randn(n)

        yield check_info_predict, J, h, J11, J21, J22, h1, h2



####################################
#  test against distribution form  #
####################################

def check_filters(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, data, inputs):
    def info_normalizer(J,h):
        out = 0.
        out += 1/2. * h.dot(np.linalg.solve(J,h))
        out -= 1/2. * np.linalg.slogdet(J)[1]
        out += h.shape[0]/2. * np.log(2*np.pi)
        return out

    ll, filtered_mus, filtered_sigmas = kalman_filter(
        mu_init, sigma_init, A, B, sigma_states, C, D, sigma_obs, inputs, data)

    py_partial_ll = info_normalizer(*dense_infoparams(
        A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, data, inputs))

    partial_ll, filtered_Js, filtered_hs = kalman_info_filter(
        *info_params(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, data, inputs))

    ll2 = partial_ll + LDSStates._extra_loglike_terms(
        A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, inputs, data)

    filtered_mus2 = [np.linalg.solve(J,h) for J, h in zip(filtered_Js, filtered_hs)]

    filtered_sigmas2 = [np.linalg.inv(J) for J in filtered_Js]

    assert all(np.allclose(mu1, mu2)
               for mu1, mu2 in zip(filtered_mus, filtered_mus2))
    assert all(np.allclose(s1, s2)
               for s1, s2 in zip(filtered_sigmas, filtered_sigmas2))

    assert np.isclose(partial_ll, py_partial_ll)

    # TODO: Log likelihood calculations are broken!
    assert np.isclose(ll, ll2)


def test_info_filter():
    for _ in range(1):
        n, p, d, T = randint(1,5), randint(1,5), 1, randint(10,20)
        model = generate_model(n,p,d)
        data, inputs = generate_data(*(model + (T,)))
        yield (check_filters,) + model + (data,inputs)


def check_info_Estep(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, inputs, data):
    ll, smoothed_mus, smoothed_sigmas, ExnxT = E_step(
        mu_init, sigma_init, A, B, sigma_states, C, D, sigma_obs, inputs, data)
    partial_ll, smoothed_mus2, smoothed_sigmas2, ExnxT2 = info_E_step(
        *info_params(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, data, inputs))

    ll2 = partial_ll + LDSStates._extra_loglike_terms(
        A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, inputs, data)

    # TODO: Log likelihood calculations are broken!
    assert np.isclose(ll,ll2)
    assert np.allclose(smoothed_mus, smoothed_mus2)
    assert np.allclose(smoothed_sigmas, smoothed_sigmas2)
    assert np.allclose(ExnxT, ExnxT2)


def test_info_Estep():
    for _ in range(5):
        n, p, d, T = randint(1, 5), randint(1, 5), 1, randint(10, 20)
        model = generate_model(n, p, d)
        data, inputs = generate_data(*(model + (T,)))
        yield (check_info_Estep,) + model + (inputs, data)


################################################
#  test extra likelihood terms in info E step  #
################################################

def extra_info_params(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, inputs, data):
    J_init = np.linalg.inv(sigma_init)
    h_init = np.linalg.solve(sigma_init, mu_init)

    logdet_pair = -np.linalg.slogdet(sigma_states)[1]
    J_yy = np.linalg.inv(sigma_obs)
    logdet_node = -np.linalg.slogdet(sigma_obs)[1]

    return J_init, h_init, logdet_pair, J_yy, logdet_node, data


def check_extra_loglike_terms(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, inputs, data):
    ex1 = LDSStates._extra_loglike_terms(
        A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, inputs, data)
    ex2 = LDSStates._info_extra_loglike_terms(
        *extra_info_params(A, B, sigma_states, C, D, sigma_obs, mu_init, sigma_init, inputs, data))

    assert np.isclose(ex1, ex2)


def test_extra_loglike():
    for _ in range(5):
        n, p, d, T = randint(1,5), randint(1,5), randint(0,3), randint(10,20)
        model = generate_model(n, p, d)
        data, inputs = generate_data(*(model + (T,)))
        yield (check_extra_loglike_terms,) + model + (inputs, data)
