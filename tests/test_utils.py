import numpy as np

from pylds.util import symm_block_tridiag_matmul, solve_symm_block_tridiag, \
    logdet_symm_block_tridiag, compute_symm_block_tridiag_covariances

def random_symm_block_tridiags(n, d):
    """
    Create a random matrix of size (n*d, n*d) with
    blocks of size d along (-1, 0, 1) diagonals
    """
    assert n > 0 and d > 0

    H_diag = np.random.rand(n, d, d)
    H_diag = np.matmul(H_diag, np.swapaxes(H_diag, -1, -2))

    H_upper_diag = np.random.rand(n-1, d, d)
    H_upper_diag = np.matmul(H_upper_diag, np.swapaxes(H_upper_diag, -1, -2))
    return H_diag, H_upper_diag

def symm_block_tridiags_to_dense(H_diag, H_upper_diag):
    n, d, _ = H_diag.shape
    H = np.zeros((n*d, n*d))
    for i in range(n):
        H[i*d:(i+1)*d, i*d:(i+1)*d] = H_diag[i]

        if i < n-1:
            H[i*d:(i+1)*d, (i+1)*d:(i+2)*d] = H_upper_diag[i]
            H[(i+1)*d:(i+2)*d, i*d:(i+1)*d] = H_upper_diag[i].T
    return H

def test_symm_block_tridiag_matmul():
    n, d = 10, 3
    for _ in range(5):
        H_diag, H_upper_diag = random_symm_block_tridiags(n, d)
        H = symm_block_tridiags_to_dense(H_diag, H_upper_diag)
        v = np.random.randn(n, d)

        out1 = H.dot(v.ravel()).reshape((n, d))
        out2 = symm_block_tridiag_matmul(H_diag, H_upper_diag, v)
        assert np.allclose(out1, out2)


def test_solve_symm_block_tridiag():
    n, d = 10, 3
    for _ in range(5):
        H_diag, H_upper_diag = random_symm_block_tridiags(n, d)
        H = symm_block_tridiags_to_dense(H_diag, H_upper_diag)

        # Make sure H is positive definite
        min_ev = np.linalg.eigvalsh(H).min()
        if min_ev < 0:
            for i in range(n):
                H_diag[i] += (-min_ev + .1) * np.eye(d)
            H += (-min_ev + .1) * np.eye(n * d)
        assert np.allclose(H, symm_block_tridiags_to_dense(H_diag, H_upper_diag))
        assert np.all(np.linalg.eigvalsh(H) > 0)

        # Make random vector to solve against
        v = np.random.randn(n, d)

        out1 = np.linalg.solve(H, v.ravel()).reshape((n, d))
        out2 = solve_symm_block_tridiag(H_diag, H_upper_diag, v)
        assert np.allclose(out1, out2)

def test_logdet_symm_block_tridiag():
    n, d = 10, 3
    for _ in range(5):
        H_diag, H_upper_diag = random_symm_block_tridiags(n, d)
        H = symm_block_tridiags_to_dense(H_diag, H_upper_diag)

        # Make sure H is positive definite
        min_ev = np.linalg.eigvalsh(H).min()
        if min_ev < 0:
            for i in range(n):
                H_diag[i] += (-min_ev + .1) * np.eye(d)
            H += (-min_ev + .1) * np.eye(n * d)
        assert np.allclose(H, symm_block_tridiags_to_dense(H_diag, H_upper_diag))
        assert np.all(np.linalg.eigvalsh(H) > 0)

        out1 = np.linalg.slogdet(H)[1]
        out2 = logdet_symm_block_tridiag(H_diag, H_upper_diag)
        assert np.allclose(out1, out2)

def test_symm_block_tridiag_covariances():
    n, d = 10, 3
    for _ in range(5):
        H_diag, H_upper_diag = random_symm_block_tridiags(n, d)
        H = symm_block_tridiags_to_dense(H_diag, H_upper_diag)

        # Make sure H is positive definite
        min_ev = np.linalg.eigvalsh(H).min()
        if min_ev < 0:
            for i in range(n):
                H_diag[i] += (-min_ev + .1) * np.eye(d)
            H += (-min_ev + .1) * np.eye(n * d)
        assert np.allclose(H, symm_block_tridiags_to_dense(H_diag, H_upper_diag))
        assert np.all(np.linalg.eigvalsh(H) > 0)

        Sigma = np.linalg.inv(H)
        sigmas, E_xtp1_xt = compute_symm_block_tridiag_covariances(H_diag, H_upper_diag)

        for i in range(n):
            assert np.allclose(Sigma[i*d:(i+1)*d, i*d:(i+1)*d], sigmas[i])

        for i in range(n-1):
            assert np.allclose(Sigma[(i+1)*d:(i+2)*d, i*d:(i+1)*d], E_xtp1_xt[i])


if __name__ == "__main__":
    test_symm_block_tridiag_matmul()
    test_solve_symm_block_tridiag()
    test_logdet_symm_block_tridiag()
    test_symm_block_tridiag_covariances()

