from __future__ import division
import numpy as np
from numpy.random import randn, randint

from pylds.lds_messages import info_predict_test, info_rts_test


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
    return out.dot(out.T)


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


##########################
#  testing info_predict  #
##########################

def py_info_predict(J,h,J11,J21,J22):
    Jnew = J + J11
    Jpredict = J22 - J21.dot(np.linalg.solve(Jnew,J21.T))
    hpredict = -J21.dot(np.linalg.solve(Jnew,h))
    lognorm = -1./2*np.linalg.slogdet(Jnew)[1] + 1./2*h.dot(np.linalg.solve(Jnew,h)) \
        - J.shape[0]/2.*np.log(2*np.pi)
    return Jpredict, hpredict, lognorm


def cy_info_predict(J,h,J11,J21,J22):
    Jpredict = np.zeros_like(J)
    hpredict = np.zeros_like(h)

    lognorm = info_predict_test(J,h,J11,J21,J22,Jpredict,hpredict)

    return Jpredict, hpredict, lognorm


def check_info_predict(J,h,J11,J21,J22):
    py_Jpredict, py_hpredict, py_lognorm = py_info_predict(J,h,J11,J21,J22)
    cy_Jpredict, cy_hpredict, cy_lognorm = cy_info_predict(J,h,J11,J21,J22)

    assert np.allclose(py_Jpredict, cy_Jpredict)
    assert np.allclose(py_hpredict, cy_hpredict)
    assert np.allclose(py_lognorm, cy_lognorm)


def test_info_predict():
    for _ in xrange(5):
        n = randint(1,20)
        J = rand_psd(n)
        h = randn(n)

        bigJ = rand_psd(2*n)
        J11, J21, J22 = map(np.copy,[bigJ[:n,:n], bigJ[n:,:n], bigJ[n:,n:]])

        yield check_info_predict, J, h, J11, J21, J22


####################################
#  testing info_rts_backward_step  #
####################################


def cy_info_rts(
        J11,J21,J22,Jpred_tp1,Jfilt_t,Jsmooth_tp1,hpred_tp1,
        hfilt_t,hsmooth_tp1):
    mu_t = np.zeros_like(hpred_tp1)
    sigma_t = np.zeros_like(J11)
    Covxnx = np.zeros_like(J21)

    info_rts_test(
        J11,J21,J22,Jpred_tp1,Jfilt_t,Jsmooth_tp1,hpred_tp1,hfilt_t,hsmooth_tp1,
        mu_t, sigma_t, Covxnx)

    return mu_t, sigma_t, Covxnx


def py_info_rts(
        J11,J21,J22,Jpred_tp1,Jfilt_t,Jsmooth_tp1,hpred_tp1,
        hfilt_t,hsmooth_tp1):
    n = J11.shape[0]
    Jtp1_future = Jsmooth_tp1 - Jpred_tp1
    htp1_future = hsmooth_tp1 - hpred_tp1
    bigJ = blockarray([[J11, J21.T], [J21, J22]]) + blockdiag([Jfilt_t, Jtp1_future])
    bigh = np.concatenate([hfilt_t, htp1_future])
    Sigma = np.linalg.inv(bigJ)
    mu = np.linalg.solve(bigJ,bigh)
    return mu[:n], Sigma[:n,:n], Sigma[:n,n:]


def check_info_rts(potentials):
    py_mu, py_sigma, py_Covxnx = py_info_rts(*potentials)
    cy_mu, cy_sigma, cy_Covxnx = cy_info_rts(*potentials)

    assert np.allclose(py_mu,cy_mu)
    assert np.allclose(py_sigma,cy_sigma)
    assert np.allclose(py_Covxnx,cy_Covxnx)


def test_info_rts():
    def generate_potentials(n):
        bigJ = rand_psd(2*n,2*n)
        J11, J21, J22 = map(np.copy,[bigJ[:n,:n], bigJ[n:,:n], bigJ[n:,n:]])

        Jnode1, Jnode2 = rand_psd(n), rand_psd(n)
        hnode1, hnode2 = randn(n), randn(n)

        Jfilt_t = J11 + Jnode1
        hfilt_t = hnode1

        Jtemp = bigJ + blockdiag([Jnode1, np.zeros((n,n))])
        htemp = np.concatenate([hnode1, np.zeros(n)])
        mu, Sigma = info_to_mean(Jtemp, htemp)
        Jpred_tp1, hpred_tp1 = mean_to_info(mu[n:], Sigma[n:,n:])

        Jtemp = bigJ + blockdiag([Jnode1, Jnode2])
        htemp = np.concatenate([hnode1, hnode2])
        mu, Sigma = info_to_mean(Jtemp, htemp)
        Jsmooth_tp1, hsmooth_tp1 = mean_to_info(mu[n:], Sigma[n:,n:])

        return J11, J21, J22, Jpred_tp1, Jfilt_t, Jsmooth_tp1, hpred_tp1, \
            hfilt_t, hsmooth_tp1

    for _ in xrange(5):
        n = randint(1,5)
        yield check_info_rts, generate_potentials(n)

