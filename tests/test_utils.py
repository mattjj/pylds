import numpy as np

from pylds.util import symm_block_tridiag_matmul, solve_symm_block_tridiag, \
    logdet_symm_block_tridiag, compute_symm_block_tridiag_covariances, \
    convert_block_tridiag_to_banded, scipy_solve_symm_block_tridiag, \
    transpose_lower_banded_matrix, scipy_sample_block_tridiag, sample_block_tridiag

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


def test_convert_block_to_banded():
    n, d = 10, 3
    for _ in range(5):
        H_diag, H_upper_diag = random_symm_block_tridiags(n, d)
        H = symm_block_tridiags_to_dense(H_diag, H_upper_diag)

        # Get the true ab matrix
        ab_true = np.zeros((2*d, n*d))
        for j in range(2*d):
            ab_true[j, :n*d-j] = np.diag(H, -j)

        ab = convert_block_tridiag_to_banded(H_diag, H_upper_diag)

        for j in range(d):
            assert np.allclose(ab_true[j], ab[j])


def test_solve_symm_block_tridiag():
    n, d = 10, 3
    for _ in range(5):
        H_diag, H_upper_diag = random_symm_block_tridiags(n, d)
        H = symm_block_tridiags_to_dense(H_diag, H_upper_diag)

        # Make sure H is positive definite
        min_ev = np.linalg.eigvalsh(H).min()
        if min_ev < 0:
            for i in range(n):
                H_diag[i] += (-min_ev + 1e-8) * np.eye(d)
            H += (-min_ev + 1e-8) * np.eye(n * d)
        assert np.allclose(H, symm_block_tridiags_to_dense(H_diag, H_upper_diag))
        assert np.all(np.linalg.eigvalsh(H) > 0)

        # Make random vector to solve against
        v = np.random.randn(n, d)

        out1 = np.linalg.solve(H, v.ravel()).reshape((n, d))
        out2 = solve_symm_block_tridiag(H_diag, H_upper_diag, v)
        out3 = scipy_solve_symm_block_tridiag(H_diag, H_upper_diag, v)
        assert np.allclose(out1, out2)
        assert np.allclose(out1, out3)


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


def test_sample_block_tridiag():
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

        # Cholesky of H
        from scipy.linalg import cholesky_banded, solve_banded
        L1 = np.linalg.cholesky(H)
        Lab = convert_block_tridiag_to_banded(H_diag, H_upper_diag)
        L2 = cholesky_banded(Lab, lower=True)
        assert np.allclose(np.diag(L1), L2[0])
        for i in range(1, 2*d):
            assert np.allclose(np.diag(L1, -i), L2[i, :-i])

        U1 = L1.T
        U2 = transpose_lower_banded_matrix(L2)
        Uab = convert_block_tridiag_to_banded(H_diag, H_upper_diag, lower=False)
        U3 = cholesky_banded(Uab, lower=False)
        assert np.allclose(np.diag(U1), U2[-1])
        assert np.allclose(np.diag(U1), U3[-1])
        for i in range(1, 2 * d):
            assert np.allclose(np.diag(U1, i), U2[-(i+1), i:])
            assert np.allclose(np.diag(U1, i), U3[-(i+1), i:])

        z = np.random.randn(n*d)
        x1 = np.linalg.solve(U1, z)
        x2 = solve_banded((0, 2*d-1), U2, z)
        x3 = scipy_sample_block_tridiag(H_diag, H_upper_diag, z=z)
        assert np.allclose(x1, x2)
        assert np.allclose(x1, x3)
        print("success")



def time_sample_block_tridiag():
    from time import time
    n, d, m = 1000, 10, 5

    ds = [5, 10, 25, 50, 100]
    ts_scipy = np.zeros_like(ds)
    ts_pylds = np.zeros_like(ds)

    for d in ds:
        print("timing test: n={} d={}".format(n, d))

        H_diag = 2 * np.eye(d)[None, :, :].repeat(n, axis=0)
        H_upper_diag = np.eye(d)[None, :, :].repeat(n-1, axis=0)

        Uab = convert_block_tridiag_to_banded(H_diag, H_upper_diag, lower=False)

        tic = time()
        for _ in range(m):
            scipy_sample_block_tridiag(H_diag, H_upper_diag)
        print("scipy:            {:.4f} sec".format((time() - tic)/m))

        tic = time()
        for _ in range(m):
            scipy_sample_block_tridiag(H_diag, H_upper_diag, ab=Uab)
        print("scipy (given ab): {:.4f} sec".format((time() - tic)/m))

        tic = time()
        for _ in range(m):
            sample_block_tridiag(H_diag, H_upper_diag)
        print("message passing:  {:.4f} sec".format((time() - tic)/m))


if __name__ == "__main__":
    test_symm_block_tridiag_matmul()
    test_convert_block_to_banded()
    test_solve_symm_block_tridiag()
    test_logdet_symm_block_tridiag()
    test_symm_block_tridiag_covariances()
    test_sample_block_tridiag()
    time_sample_block_tridiag()

