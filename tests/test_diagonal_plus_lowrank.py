from __future__ import division
import numpy as np
from numpy.random import rand, randn, randint
from scipy.stats import multivariate_normal

from pylds.lds_messages import test_solve_diagonal_plus_lowrank, test_condition_on_diagonal
from pylds.lds_messages_interface import filter_and_sample, filter_and_sample_diagonal
from test_infofilter import generate_data, spectral_radius


##########
#  util  #
##########

def rand_psd(n,k=None):
    k = k if k else n
    out = randn(n,k)
    return np.atleast_2d(out.dot(out.T))


def generate_diag_model(n,p):
    A = randn(n,n)
    A /= 2.*spectral_radius(A)  # ensures stability
    assert spectral_radius(A) < 1.

    B = randn(n,n)
    C = randn(p,n)
    d = rand(p)

    mu_init = randn(n)
    sigma_init = rand_psd(n)

    return A, B, C, d, mu_init, sigma_init


###########
#  tests  #
###########

# test solve_diagonal_plus_lowrank

def solve_diagonal_plus_lowrank(a,B,C,b):
    out = b.copy(order='F')
    B = np.asfortranarray(B)
    C_is_identity = np.allclose(C,np.eye(C.shape[0]))
    logdet = test_solve_diagonal_plus_lowrank(a,B,C,C_is_identity,out)
    return logdet, out


def check_diagonal_plus_lowrank(a,B,C,b):
    solve1 = np.linalg.solve(np.diag(a)+B.dot(C).dot(B.T), b)
    logdet1 = np.linalg.slogdet(np.diag(a)+B.dot(C).dot(B.T))[1]
    logdet2, solve2 = solve_diagonal_plus_lowrank(a,B,C,b)

    assert np.isclose(logdet1, logdet2)
    assert np.allclose(solve1, solve2)


def test_cython_diagonal_plus_lowrank():
    for _ in xrange(5):
        n, p, k = randint(1,10), randint(1,10), randint(1,10)
        a = rand(p)
        B = randn(p,n)
        b = randn(p,k)
        C = np.eye(n) if rand() < 0.5 else rand_psd(n)

        yield check_diagonal_plus_lowrank, a, B, C, b


# test condition_on_diagonal

def cython_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y):
    mu_cond = np.random.randn(*mu_x.shape)
    sigma_cond = np.random.randn(*sigma_x.shape)
    ll = test_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y, mu_cond, sigma_cond)
    return ll, mu_cond, sigma_cond


def check_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y):
    def condition_on(mu_x, sigma_x, C, sigma_obs, y):
        p, n = C.shape
        sigma_xy = sigma_x.dot(C.T)
        sigma_yy = C.dot(sigma_x).dot(C.T) + np.diag(sigma_obs)
        mu_y = C.dot(mu_x)
        mu = mu_x + sigma_xy.dot(np.linalg.solve(sigma_yy, y - mu_y))
        sigma = sigma_x - sigma_xy.dot(np.linalg.solve(sigma_yy,sigma_xy.T))

        ll = multivariate_normal.logpdf(y,mu_y,sigma_yy)

        return ll, mu, sigma

    py_ll, py_mu, py_sigma = condition_on(mu_x, sigma_x, C, sigma_obs, y)
    cy_ll, cy_mu, cy_sigma = cython_condition_on_diagonal(mu_x, sigma_x, C, sigma_obs, y)

    assert np.allclose(py_mu, cy_mu)
    assert np.allclose(py_sigma, cy_sigma)
    assert np.isclose(py_ll, cy_ll)


def test_cython_condition_on_diagonal():
    for _ in xrange(5):
        n, p = randint(1,10), randint(1,10)
        mu_x = randn(n)
        sigma_x = rand_psd(n)
        C = randn(p,n)
        sigma_obs = rand(p)
        y = randn(p)

        yield check_condition_on_diagonal, mu_x, sigma_x, C, sigma_obs, y


# test filter_and_sample

def check_filter_and_sample(A, B, C, d, mu_init, sigma_init, data):
    rngstate = np.random.get_state()
    ll1, sample1 = filter_and_sample(
        mu_init, sigma_init, A, B.dot(B.T), C, np.diag(d**2), data)
    np.random.set_state(rngstate)
    ll2, sample2 = filter_and_sample_diagonal(
        mu_init, sigma_init, A, B.dot(B.T), C, d**2, data)

    assert np.isclose(ll1, ll2)
    assert np.allclose(sample1, sample2)


def test_filter_and_sample():
    for _ in xrange(5):
        n, p, T = randint(1,5), randint(1,5), randint(10,20)
        model = generate_diag_model(n,p)
        data = generate_data(*(model + (T,)))
        yield (check_filter_and_sample,) + model + (data,)


