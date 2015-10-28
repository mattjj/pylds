from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal

from pylds.models import DefaultLDS


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
    n, T = model.n, model.states_list[0].T

    dense_mu = np.linalg.solve(J,h).reshape((T,n))

    model.E_step()
    model_mu = model.states_list[0].smoothed_mus

    assert np.allclose(dense_mu,model_mu)


def same_marginal_covs(model,(J,h)):
    n, T = model.n, model.states_list[0].T

    all_dense_sigmas = np.linalg.inv(J)
    dense_sigmas = np.array([all_dense_sigmas[k*n:(k+1)*n,k*n:(k+1)*n]
                             for k in range(T)])

    model.E_step()
    model_sigmas = model.states_list[0].smoothed_sigmas

    assert np.allclose(dense_sigmas,model_sigmas)


def same_pairwise_secondmoments(model,(J,h)):
    n, T = model.n, model.states_list[0].T

    all_dense_sigmas = np.linalg.inv(J)
    dense_mu = np.linalg.solve(J,h)
    blockslices = [slice(k*n,(k+1)*n) for k in range(T)]
    dense_Extp1_xtT = \
        sum(all_dense_sigmas[tp1,t] + np.outer(dense_mu[tp1],dense_mu[t])
            for tp1,t in zip(blockslices[1:],blockslices[:-1]))

    model.E_step()
    model_Extp1_xtT = model.states_list[0].E_dynamics_stats[1]

    assert np.allclose(dense_Extp1_xtT,model_Extp1_xtT)


def same_loglike(model,_):
    # NOTE: ignore the posterior (J,h) passed in so we can use the more
    # convenient prior info parameters
    data = model.states_list[0].data
    T = data.shape[0]

    C, model.C = model.C, np.zeros_like(model.C)
    J,h = lds_to_dense_infoparams(model,data)
    model.C = C

    bigC = np.kron(np.eye(T),C)
    mu_x = np.linalg.solve(J,h)
    sigma_x = np.linalg.inv(J)
    mu_y = bigC.dot(mu_x)
    sigma_y = bigC.dot(sigma_x).dot(bigC.T) + np.kron(np.eye(T),model.sigma_obs)
    dense_loglike = multivariate_normal.logpdf(data.ravel(),mu_y,sigma_y)

    model_loglike = model.log_likelihood()

    assert np.isclose(dense_loglike, model_loglike)


def random_model(n,p,T):
    data = np.random.randn(T,p)
    model = DefaultLDS(n,p)
    model.A = 0.99*random_rotation(n,0.01)
    model.C = np.random.randn(p,n)

    J,h = lds_to_dense_infoparams(model,data)
    model.add_data(data)

    return model, (J,h)


def check_random_model(check):
    n, p = np.random.randint(2,5), np.random.randint(2,5)
    T = np.random.randint(10,20)
    check(*random_model(n,p,T))


def test_means():
    for _ in range(5):
        yield check_random_model, same_means


def test_marginals_covs():
    for _ in range(5):
        yield check_random_model, same_marginal_covs


def test_pairwise_secondmoments():
    for _ in range(5):
        yield check_random_model, same_pairwise_secondmoments


def test_loglike():
    for _ in range(5):
        yield check_random_model, same_loglike
