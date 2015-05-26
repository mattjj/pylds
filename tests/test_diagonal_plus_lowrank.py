from __future__ import division
import numpy as np
from numpy.random import rand, randn, randint

from pylds.lds_messages import test_solve_diagonal_plus_lowrank, test_condition_on_diagonal


##########
#  util  #
##########

def rand_psd(n,k=None):
    k = k if k else n
    out = randn(n,k)
    return np.atleast_2d(out.dot(out.T))


###########
#  tests  #
###########

def solve_diagonal_plus_lowrank(a,B,C,b):
    out = b.copy(order='F')
    B = np.asfortranarray(B)
    C_is_identity = np.allclose(C,np.eye(C.shape[0]))
    test_solve_diagonal_plus_lowrank(a,B,C,C_is_identity,out)
    return out


def check_diagonal_plus_lowrank(a,B,C,b):
    solve1 = np.linalg.solve(np.diag(a)+B.dot(C).dot(B.T), b)
    solve2 = solve_diagonal_plus_lowrank(a,B,C,b)

    assert np.allclose(solve1,solve2)


def test_cython_diagonal_plus_lowrank():
    for _ in xrange(5):
        n, p, k = randint(1,10), randint(1,10), randint(1,10)
        a = rand(p)
        B = randn(p,n)
        b = randn(p,k)
        C = np.eye(n) if rand() < 0.5 else rand_psd(n)

        yield check_diagonal_plus_lowrank, a, B, C, b


def cython_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y):
    mu_cond = np.random.randn(*mu_x.shape)
    sigma_cond = np.random.randn(*sigma_x.shape)
    test_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y, mu_cond, sigma_cond)
    return mu_cond, sigma_cond


def check_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y):
    def condition_on(mu_x, sigma_x, C, sigma_obs, y):
        sigma_xy = sigma_x.dot(C.T)
        sigma_yy = C.dot(sigma_x).dot(C.T) + np.diag(sigma_obs)
        mu_y = C.dot(mu_x)
        mu = mu_x + sigma_xy.dot(np.linalg.solve(sigma_yy, y - mu_y))
        sigma = sigma_x - sigma_xy.dot(np.linalg.solve(sigma_yy,sigma_xy.T))
        return mu, sigma

    py_mu, py_sigma = condition_on(mu_x, sigma_x, C, sigma_obs, y)
    cy_mu, cy_sigma = cython_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y)

    assert np.allclose(py_mu, cy_mu)
    assert np.allclose(py_sigma, cy_sigma)


def test_cython_condition_on_diagonal():
    for _ in xrange(5):
        n, p = randint(1,10), randint(1,10)
        mu_x = randn(n)
        sigma_x = rand_psd(n)
        C = randn(p,n)
        sigma_obs = rand(p)
        y = randn(p)

        yield check_condition_on_diagonal, mu_x, sigma_x, C, sigma_obs, y


