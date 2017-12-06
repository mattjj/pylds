import autograd.numpy as np

from pylds.lds_messages_interface import info_E_step, kalman_info_filter, info_sample

def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def symm_block_tridiag_matmul(H_diag, H_upper_diag, v):
    """
    Compute matrix-vector product with a symmetric block
    tridiagonal matrix H and vector v.

    :param H_diag:          block diagonal terms of H
    :param H_upper_diag:    upper block diagonal terms of H
    :param v:               vector to multiple
    :return:                H * v
    """
    T, D, _ = H_diag.shape
    assert H_diag.ndim == 3 and H_diag.shape[2] == D
    assert H_upper_diag.shape == (T-1, D, D)
    assert v.shape == (T, D)

    out = np.matmul(H_diag, v[:, :, None])[:, :, 0]
    out[:-1] += np.matmul(H_upper_diag, v[1:][:, :, None])[:, :, 0]
    out[1:] += np.matmul(np.swapaxes(H_upper_diag, -2, -1), v[:-1][:, :, None])[:, :, 0]
    return out


def solve_symm_block_tridiag(H_diag, H_upper_diag, v):
    """
    use the info smoother to solve a symmetric block tridiagonal system
    """
    T, D, _ = H_diag.shape
    assert H_diag.ndim == 3 and H_diag.shape[2] == D
    assert H_upper_diag.shape == (T - 1, D, D)
    assert v.shape == (T, D)

    J_init = J_11 = J_22 = np.zeros((D, D))
    h_init = h_1 = h_2 = np.zeros((D,))

    J_21 = np.swapaxes(H_upper_diag, -1, -2)
    J_node = H_diag
    h_node = v

    _, y, _, _ = info_E_step(J_init, h_init, 0,
                             J_11, J_21, J_22, h_1, h_2, np.zeros((T-1)),
                             J_node, h_node, np.zeros(T))
    return y


def convert_block_tridiag_to_banded(H_diag, H_upper_diag, lower=True):
    """
    convert blocks to banded matrix representation required for scipy.
    we are using the "lower form."
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    T, D, _ = H_diag.shape
    assert H_diag.ndim == 3 and H_diag.shape[2] == D
    assert H_upper_diag.shape == (T - 1, D, D)
    H_lower_diag = np.swapaxes(H_upper_diag, -2, -1)

    ab = np.zeros((2 * D, T * D))

    # Fill in blocks along the diagonal
    for d in range(D):
        # Get indices of (-d)-th diagonal of H_diag
        i = np.arange(d, D)
        j = np.arange(0, D - d)
        h = np.column_stack((H_diag[:, i, j], np.zeros((T, d))))
        ab[d] = h.ravel()

    # Fill in lower left corner of blocks below the diagonal
    for d in range(0, D):
        # Get indices of (-d)-th diagonal of H_diag
        i = np.arange(d, D)
        j = np.arange(0, D - d)
        h = np.column_stack((H_lower_diag[:, i, j], np.zeros((T - 1, d))))
        ab[D + d, :D * (T - 1)] = h.ravel()

    # Fill in upper corner of blocks below the diagonal
    for d in range(1, D):
        # Get indices of (+d)-th diagonal of H_lower_diag
        i = np.arange(0, D - d)
        j = np.arange(d, D)
        h = np.column_stack((np.zeros((T - 1, d)), H_lower_diag[:, i, j]))
        ab[D - d, :D * (T - 1)] += h.ravel()

    return ab if lower else transpose_lower_banded_matrix(ab)


def transpose_lower_banded_matrix(Lab):
    # This is painful
    Uab = np.flipud(Lab)
    u = Uab.shape[0] - 1
    for i in range(1,u+1):
        Uab[-(i+1), i:] = Uab[-(i+1), :-i]
        Uab[-(i + 1), :i] = 0
    return Uab


def scipy_solve_symm_block_tridiag(H_diag, H_upper_diag, v, ab=None):
    """
    use scipy.linalg.solve_banded to solve a symmetric block tridiagonal system

    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    from scipy.linalg import solveh_banded
    ab = convert_block_tridiag_to_banded(H_diag, H_upper_diag) \
        if ab is None else ab
    x = solveh_banded(ab, v.ravel(), lower=True)
    return x.reshape(v.shape)


def scipy_sample_block_tridiag(H_diag, H_upper_diag, size=1, ab=None, z=None):
    from scipy.linalg import cholesky_banded, solve_banded

    ab = convert_block_tridiag_to_banded(H_diag, H_upper_diag, lower=False) \
        if ab is None else ab

    Uab = cholesky_banded(ab, lower=False)
    z = np.random.randn(ab.shape[1], size) if z is None else z

    # If lower = False, we have (U^T U)^{-1} = U^{-1} U^{-T} = AA^T = Sigma
    # where A = U^{-1}.  Samples are Az = U^{-1}z = x, or equivalently Ux = z.
    return solve_banded((0, Uab.shape[0]-1), Uab, z)


def sample_block_tridiag(H_diag, H_upper_diag):
    """
    helper function for sampling block tridiag gaussians.
    this is only for speed comparison with the solve approach.
    """
    T, D, _ = H_diag.shape
    assert H_diag.ndim == 3 and H_diag.shape[2] == D
    assert H_upper_diag.shape == (T - 1, D, D)

    J_init = J_11 = J_22 = np.zeros((D, D))
    h_init = h_1 = h_2 = np.zeros((D,))

    J_21 = np.swapaxes(H_upper_diag, -1, -2)
    J_node = H_diag
    h_node = np.zeros((T,D))

    y = info_sample(J_init, h_init, 0,
                    J_11, J_21, J_22, h_1, h_2, np.zeros((T-1)),
                    J_node, h_node, np.zeros(T))
    return y


def logdet_symm_block_tridiag(H_diag, H_upper_diag):
    """
    compute the log determinant of a positive definite,
    symmetric block tridiag matrix.  Use the Kalman
    info filter to do so.  Specifically, the KF computes
    the normalizer:

        log Z = 1/2 h^T J^{-1} h -1/2 log |J| +n/2 log 2 \pi

    We set h=0 to get -1/2 log |J| + n/2 log 2 \pi and from
    this we solve for log |J|.
    """
    T, D, _ = H_diag.shape
    assert H_diag.ndim == 3 and H_diag.shape[2] == D
    assert H_upper_diag.shape == (T - 1, D, D)

    J_init = J_11 = J_22 = np.zeros((D, D))
    h_init = h_1 = h_2 = np.zeros((D,))
    log_Z_init = 0

    J_21 = np.swapaxes(H_upper_diag, -1, -2)
    log_Z_pair = 0

    J_node = H_diag
    h_node = np.zeros((T, D))
    log_Z_node = 0

    logZ, _, _ = kalman_info_filter(J_init, h_init, log_Z_init,
                                    J_11, J_21, J_22, h_1, h_2, log_Z_pair,
                                    J_node, h_node, log_Z_node)

    # logZ = -1/2 log |J| + n/2 log 2 \pi
    logdetJ = -2 * (logZ - (T*D) / 2 * np.log(2 * np.pi))
    return logdetJ


def compute_symm_block_tridiag_covariances(H_diag, H_upper_diag):
    """
    use the info smoother to solve a symmetric block tridiagonal system
    """
    T, D, _ = H_diag.shape
    assert H_diag.ndim == 3 and H_diag.shape[2] == D
    assert H_upper_diag.shape == (T - 1, D, D)

    J_init = J_11 = J_22 = np.zeros((D, D))
    h_init = h_1 = h_2 = np.zeros((D,))

    J_21 = np.swapaxes(H_upper_diag, -1, -2)
    J_node = H_diag
    h_node = np.zeros((T, D))

    _, _, sigmas, E_xt_xtp1 = \
        info_E_step(J_init, h_init, 0,
                    J_11, J_21, J_22, h_1, h_2, np.zeros((T-1)),
                    J_node, h_node, np.zeros(T))
    return sigmas, E_xt_xtp1
