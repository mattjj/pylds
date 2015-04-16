from __future__ import division
import numpy as np

from pybasicbayes.distributions import Regression
from autoregressive.distributions import AutoRegression

from models import LDS, DefaultLDS

np.random.seed(0)


##########
#  util  #
##########

def cumsum(v,strict=False):
    if not strict:
        return np.cumsum(v,axis=0)
    else:
        out = np.zeros_like(v)
        out[1:] = np.cumsum(v[:-1],axis=0)
        return out


def bmat(blocks):
    rowsizes = [row[0].shape[0] for row in blocks]
    colsizes = [col[0].shape[1] for col in zip(*blocks)]
    rowstarts = cumsum(rowsizes,strict=True)
    colstarts = cumsum(colsizes,strict=True)

    nrows, ncols = sum(rowsizes), sum(colsizes)
    out = np.zeros((nrows,ncols))

    for i, (rstart, rsz) in enumerate(zip(rowstarts, rowsizes)):
        for j, (cstart, csz) in enumerate(zip(colstarts, colsizes)):
            out[rstart:rstart+rsz,cstart:cstart+csz] = blocks[i][j]

    return out


def random_rotation(n,theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n,n))
    out[:2,:2] = rot
    q = np.linalg.qr(np.random.randn(n,n))[0]
    return q.dot(out).dot(q.T)


def lds_to_dense_infoparams(model,data):
    T, n = data.shape[0], model.n

    mu_init, sigma_init = model.mu_init, model.sigma_init
    A, sigma_states = model.A, model.sigma_states
    C, sigma_obs = model.C, model.sigma_obs

    h = C.T.dot(np.linalg.solve(sigma_obs, data.T)).T
    h[0] += np.linalg.solve(sigma_init, mu_init)

    J = np.kron(np.eye(T),C.T.dot(np.linalg.solve(sigma_obs,C)))
    J[:n,:n] += np.linalg.inv(sigma_init)
    ss_inv = np.linalg.inv(sigma_states)
    pairblock = bmat([[A.T.dot(ss_inv).dot(A), -A.T.dot(ss_inv)],
                      [-ss_inv.dot(A), ss_inv]])
    for t in range(0,n*(T-1),n):
        J[t:t+2*n,t:t+2*n] += pairblock

    return J.reshape(T*n,T*n), h.reshape(T*n)


###########
#  tests  #
###########

def same_means(model,(J,h)):
    n,p,T = model.n, model.p, model.states_list[0].T
    dense_mu = np.linalg.solve(J,h).reshape((T,n))
    model_mu = model.states_list[0].smoothed_mus
    return np.allclose(dense_mu,model_mu)


def same_marginal_covs(model,(J,h)):
    n,p,T = model.n, model.p, model.states_list[0].T
    all_dense_sigmas = np.linalg.inv(J)
    dense_sigmas = np.array([all_dense_sigmas[k*n:(k+1)*n,k*n:(k+1)*n]
                             for k in range(T)])
    model_sigmas = model.states_list[0].smoothed_sigmas
    return np.allclose(dense_sigmas,model_sigmas)


def test_random():
    def helper(n,p,T):
        data = np.random.randn(T,p)
        model = DefaultLDS(n,p)
        model.A = 0.99*random_rotation(n,0.01)
        model.C = np.random.randn(p,n)

        J,h = lds_to_dense_infoparams(model,data)
        model.add_data(data).E_step()

        assert same_means(model,(J,h))
        assert same_marginal_covs(model,(J,h))

    for _ in range(3):
        yield helper, np.random.randint(2,5), np.random.randint(2,5), 10

