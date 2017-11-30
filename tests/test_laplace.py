import numpy as np
from pylds.models import DefaultLDS
from pylds.laplace import LaplaceApproxPoissonLDSStates


def random_states(T, N, D):
    model = DefaultLDS(N, D, D_input=0, sigma_states=1e8 * np.eye(D))
    data = np.random.poisson(3.0, size=(T, N))
    states = LaplaceApproxPoissonLDSStates(model, data=data)
    states.gaussian_states *= 0
    return states

def test_gradient(T=100, N=10, D=3):
    states = random_states(T, N, D)
    x = np.random.randn(T, D)
    g_sparse = states.gradient_log_joint(x)
    g_full = states.test_gradient_log_joint(x)
    assert np.allclose(g_sparse, g_full)


def test_hessian(T=100, N=10, D=3):
    states = random_states(T, N, D)
    x = np.random.randn(T, D)
    H_diag, H_upper_diag = states.sparse_hessian_log_joint(x)
    H_full = states.test_hessian_log_joint(x)

    for t in range(T):
        assert np.allclose(H_full[t,:,t,:], H_diag[t])
        if t < T - 1:
            assert np.allclose(H_full[t,:,t+1,:], H_upper_diag[t])
            assert np.allclose(H_full[t+1,:,t,:], H_upper_diag[t].T)


def test_hessian_vector_product(T=100, N=10, D=3):
    states = random_states(T, N, D)
    x = np.random.randn(T, D)
    H_full = states.test_hessian_log_joint(x)
    v = np.random.randn(T, D)

    hvp1 = H_full.reshape((T*D, T*D)).dot(v.ravel())
    hvp2 = states.hessian_vector_product_log_joint(x, v)
    assert np.allclose(hvp1.reshape((T, D)), hvp2)


def test_laplace_approximation(T=50000, N=100, D=10):
    states = random_states(T, N, D)
    xhat = states.laplace_approximation(method="bfgs", verbose=True)
    g = states.gradient_log_joint(xhat)
    assert np.allclose(g, 0, atol=1e-2)


def test_laplace_approximation_secondorder(T=50000, N=100, D=10):
    states = random_states(T, N, D)
    xhat = states.laplace_approximation(method="newton", stepsz=.99, verbose=True)
    g = states.gradient_log_joint(xhat)
    assert np.allclose(g, 0, atol=1e-2)

if __name__ == "__main__":
    test_gradient()
    test_hessian()
    test_hessian_vector_product()
    test_laplace_approximation(T=1000, N=20, D=5)
    test_laplace_approximation_secondorder(T=50000, N=100, D=10)

