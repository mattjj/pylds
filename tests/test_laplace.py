import numpy as np
from pylds.models import DefaultPoissonLDS, DefaultBernoulliLDS
from pylds.laplace import LaplaceApproxPoissonLDSStates, LaplaceApproxBernoulliLDSStates

from nose.tools import nottest


def correct_gradient(states):
    x = np.random.randn(states.T, states.D_latent)
    g_sparse = states.gradient_log_joint(x)
    g_full = states.test_gradient_log_joint(x)
    assert np.allclose(g_sparse, g_full)


def correct_hessian(states):
    x = np.random.randn(states.T, states.D_latent)
    H_diag, H_upper_diag = states.sparse_hessian_log_joint(x)
    H_full = states.test_hessian_log_joint(x)

    for t in range(states.T):
        assert np.allclose(H_full[t,:,t,:], H_diag[t])
        if t < states.T - 1:
            assert np.allclose(H_full[t,:,t+1,:], H_upper_diag[t])
            assert np.allclose(H_full[t+1,:,t,:], H_upper_diag[t].T)


def correct_hessian_vector_product(states):
    T, D = states.T, states.D_latent
    x = np.random.randn(T, D)
    H_full = states.test_hessian_log_joint(x)
    v = np.random.randn(T, D)

    hvp1 = H_full.reshape((T*D, T*D)).dot(v.ravel())
    hvp2 = states.hessian_vector_product_log_joint(x, v)
    assert np.allclose(hvp1.reshape((T, D)), hvp2)


def correct_laplace_approximation_bfgs(states):
    xhat = states.laplace_approximation(method="bfgs", verbose=True)
    g = states.gradient_log_joint(xhat)
    assert np.allclose(g, 0, atol=1e-2)

def correct_laplace_approximation_newton(states):
    xhat = states.laplace_approximation(method="newton", verbose=True)
    g = states.gradient_log_joint(xhat)
    assert np.allclose(g, 0, atol=1e-2)

@nottest
def test_laplace_approximation_newton_largescale():
    T = 50000
    N = 100
    D = 10
    D_input = 1
    model = DefaultPoissonLDS(N, D, D_input=D_input)
    data = np.random.poisson(3.0, size=(T, N))
    inputs = np.random.randn(T, D_input)
    states = LaplaceApproxPoissonLDSStates(model, data=data, inputs=inputs)
    states.gaussian_states *= 0

    xhat = states.laplace_approximation(method="newton", stepsz=.99, verbose=True)
    g = states.gradient_log_joint(xhat)
    assert np.allclose(g, 0, atol=1e-2)


def check_random_poisson_states(check):
    T = np.random.randint(25, 200)
    N = np.random.randint(10, 20)
    D = np.random.randint(1, 10)
    D_input = np.random.randint(0, 2)

    model = DefaultPoissonLDS(N, D, D_input=D_input)
    data = np.random.poisson(3.0, size=(T, N))
    inputs = np.random.randn(T, D_input)
    states = LaplaceApproxPoissonLDSStates(model, data=data, inputs=inputs)
    states.gaussian_states *= 0

    check(states)


def check_random_bernoulli_states(check):
    T = np.random.randint(25, 200)
    N = np.random.randint(10, 20)
    D = np.random.randint(1, 10)
    D_input = np.random.randint(0, 2)

    model = DefaultBernoulliLDS(N, D, D_input=D_input)
    data = np.random.rand(T, N)
    inputs = np.random.randn(T, D_input)
    states = LaplaceApproxBernoulliLDSStates(model, data=data, inputs=inputs)
    states.gaussian_states *= 0

    check(states)


### Poisson tests
def test_poisson_gradients():
    for _ in range(5):
        yield check_random_poisson_states, correct_gradient


def test_poisson_hessian():
    for _ in range(5):
        yield check_random_poisson_states, correct_hessian


def test_poisson_hessian_vector_product():
    for _ in range(5):
        yield check_random_poisson_states, correct_hessian_vector_product


def test_poisson_laplace_approximation_bfgs():
    for _ in range(5):
        yield check_random_poisson_states, correct_laplace_approximation_bfgs


def test_poisson_laplace_approximation_newton():
    for _ in range(5):
        yield check_random_poisson_states, correct_laplace_approximation_newton


### Bernoulli tests
def test_bernoulli_gradients():
    for _ in range(5):
        yield check_random_bernoulli_states, correct_gradient


def test_bernoulli_hessian():
    for _ in range(5):
        yield check_random_bernoulli_states, correct_hessian


def test_bernoulli_hessian_vector_product():
    for _ in range(5):
        yield check_random_bernoulli_states, correct_hessian_vector_product


def test_bernoulli_laplace_approximation_bfgs():
    for _ in range(5):
        yield check_random_bernoulli_states, correct_laplace_approximation_bfgs


def test_bernoulli_laplace_approximation_newton():
    for _ in range(5):
        yield check_random_bernoulli_states, correct_laplace_approximation_newton
