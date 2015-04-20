from __future__ import division
import numpy as np
import numpy.random as npr

from pylds.lds_messages import info_predict_test


def rand_psd(n,k=None):
    k = k if k else n
    out = npr.randn(n,k)
    return out.dot(out.T)


def py_info_predict(J,h,J11,J12,J22):
    Jnew = J + J11
    Jpredict = J22 - J12.T.dot(np.linalg.solve(Jnew,J12))
    hpredict = -J12.T.dot(np.linalg.solve(Jnew,h))
    lognorm = -1./2*np.linalg.slogdet(Jnew)[1] + 1./2*h.dot(np.linalg.solve(Jnew,h)) \
        - J.shape[0]/2.*np.log(2*np.pi)
    return Jpredict, hpredict, lognorm


def cy_info_predict(J,h,J11,J12,J22):
    Jpredict = np.zeros_like(J)
    hpredict = np.zeros_like(h)

    lognorm = info_predict_test(J,h,J11,J12,J22,Jpredict,hpredict)

    return Jpredict, hpredict, lognorm


def check_info_predict(J,h,J11,J12,J22):
    py_Jpredict, py_hpredict, py_lognorm = py_info_predict(J,h,J11,J12,J22)
    cy_Jpredict, cy_hpredict, cy_lognorm = cy_info_predict(J,h,J11,J12,J22)

    assert np.allclose(py_Jpredict, cy_Jpredict)
    assert np.allclose(py_hpredict, cy_hpredict)
    assert np.allclose(py_lognorm, cy_lognorm)


def test_info_predict():
    for _ in xrange(5):
        n = npr.randint(1,20)
        J = rand_psd(n)
        h = npr.randn(n)

        bigJ = rand_psd(2*n)
        J11, J12, J22 = map(np.copy,[bigJ[:n,:n], bigJ[:n,n:], bigJ[n:,n:]])

        yield check_info_predict, J, h, J11, J12, J22

