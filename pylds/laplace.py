import autograd.numpy as np
from autograd.scipy.special import gammaln
from autograd import grad, hessian

from pylds.states import _LDSStates
from pylds.util import symm_block_tridiag_matmul
from pylds.lds_messages_interface import info_E_step


class _LaplaceApproxLDSStatesBase(_LDSStates):
    """
    Support variational inference via Laplace approximation.
    The key is a definition of the log likelihood,

        log p(y_t | x_t, \theta)

    Combining this with a Gaussian LDS prior on the states,
    we can compute the gradient and Hessian of the log likelihood.
    """
    def local_log_likelihood(self, xt, yt, ut):
        """
        Return log p(yt | xt).  Implement this in base classes.
        """
        raise NotImplementedError

    def log_conditional_likelihood(self, x):
        """
        likelihood \sum_t log p(y_t | x_t)
        Optionally override this in base classes
        """
        T, D = self.T, self.D_latent
        assert x.shape == (T, D)

        ll = 0
        for t in range(self.T):
            ll += self.local_log_likelihood(x[t], self.data[t], self.inputs[t])
        return ll

    def grad_local_log_likelihood(self, x):
        """
        return d/dxt log p(yt | xt)  evaluated at xt
        Optionally override this in base classes
        """
        T, D = self.T, self.D_latent
        assert x.shape == (T, D)
        gfun = grad(self.local_log_likelihood)

        g = np.zeros((T, D))
        for t in range(T):
            g[t] += gfun(x[t], self.data[t], self.inputs[t])
        return g

    def hessian_local_log_likelihood(self, x):
        """
        return d^2/dxt^2 log p(y | x) for each time bin
        Optionally override this in base classes
        """
        T, D = self.T, self.D_latent
        assert x.shape == (T, D)

        hfun = hessian(self.local_log_likelihood)
        H_diag = np.zeros((T, D, D))
        for t in range(T):
            H_diag[t] = hfun(x[t], self.data[t], self.inputs[t])
        return H_diag

    @property
    def sparse_J_prior(self):
        T, D = self.T, self.D_latent
        J_init, _, _ = self.info_init_params
        J_11, J_21, J_22, _, _, _ = self.info_dynamics_params

        # Collect the Gaussian LDS prior terms
        J_diag = np.zeros((T, D, D))
        J_diag[0] += J_init
        J_diag[:-1] += J_11
        J_diag[1:] += J_22

        J_upper_diag = np.repeat(J_21.T[None, :, :], T - 1, axis=0)
        return J_diag, J_upper_diag

    def log_joint(self, x):
        """
        Compute the log joint probability p(x, y)
        """
        T, D = self.T, self.D_latent
        assert x.shape == (T, D)

        # prior log p(x) -- quadratic terms
        J_diag, J_upper_diag = self.sparse_J_prior
        lp = -0.5 * np.sum(x * symm_block_tridiag_matmul(J_diag, J_upper_diag, x))

        # prior log p(x) -- linear terms
        _, h_init, log_Z_init = self.info_init_params
        _, _, _, h1, h2, log_Z_dyn = self.info_dynamics_params
        lp += x[0].dot(h_init)
        lp += np.sum(x[:-1] * h1)
        lp += np.sum(x[1:] * h2)

        # prior log p(x) -- normalization constants
        lp += log_Z_init
        lp += np.sum(log_Z_dyn)

        # likelihood log p(y | x)
        lp += self.log_conditional_likelihood(x)

        return lp

    def sparse_hessian_log_joint(self, x):
        """
        The Hessian includes the quadratic terms of the Gaussian LDS prior
        as well as the Hessian of the local log likelihood.
        """
        T, D = self.T, self.D_latent
        assert x.shape == (T, D)

        # Collect the Gaussian LDS prior terms
        J_diag, J_upper_diag = self.sparse_J_prior
        H_diag, H_upper_diag = -J_diag, -J_upper_diag

        # Collect the likelihood terms
        H_diag += self.hessian_local_log_likelihood(x)

        # Subtract a little bit to ensure negative definiteness
        H_diag -= 1e-8 * np.eye(D)

        return H_diag, H_upper_diag

    def hessian_vector_product_log_joint(self, x, v):
        H_diag, H_upper_diag = self.sparse_hessian_log_joint(x)
        return symm_block_tridiag_matmul(H_diag, H_upper_diag, v)

    def gradient_log_joint(self, x):
        """
        The gradient of the log joint probability.

        For the Gaussian terms, this is

            d/dx [-1/2 x^T J x + h^T x] = -Jx + h.

        For the likelihood terms, we have for each time t

            d/dx log p(yt | xt)
        """
        T, D = self.T, self.D_latent
        assert x.shape == (T, D)

        # Collect the Gaussian LDS prior terms
        _, h_init, _ = self.info_init_params
        _, _, _, h1, h2, _ = self.info_dynamics_params
        H_diag, H_upper_diag = self.sparse_J_prior

        # Compute the gradient from the prior
        g = -1 * symm_block_tridiag_matmul(H_diag, H_upper_diag, x)
        g[0] += h_init
        g[:-1] += h1
        g[1:] += h2

        # Compute gradient from the likelihood terms
        g += self.grad_local_log_likelihood(x)

        return g

    def laplace_approximation(self, method="newton", verbose=False, tol=1e-7, **kwargs):
        if method.lower() == "newton":
            return self._laplace_approximation_newton(verbose=verbose, tol=tol, **kwargs)
        elif method.lower() == "bfgs":
            return self._laplace_approximation_bfgs(verbose=verbose, tol=tol, **kwargs)
        else:
            raise Exception("Invalid method: {}".format(method))

    def _laplace_approximation_bfgs(self, tol=1e-7, verbose=False):
        from scipy.optimize import minimize

        # Gradient ascent on the log joint probability to get mu
        T, D = self.T, self.D_latent
        scale = self.T * self.D_emission
        obj = lambda xflat: -self.log_joint(xflat.reshape((T, D))) / scale
        jac = lambda xflat: -self.gradient_log_joint(xflat.reshape((T, D))).ravel() / scale
        hvp = lambda xflat, v: -self.hessian_vector_product_log_joint(
            xflat.reshape((T, D)), v.reshape((T, D))).ravel() / scale

        x0 = self.gaussian_states.reshape((T * D,))

        # Make callback
        itr = [0]

        def cbk(x):
            print("Iteration: ", itr[0],
                  "\tObjective: ", obj(x).round(2),
                  "\tAvg Grad: ", jac(x).mean().round(2))
            itr[0] += 1

        # Second order method
        if verbose:
            print("Fitting Laplace approximation")

        res = minimize(obj, x0,
                       tol=tol,
                       method="Newton-CG",
                       jac=jac,
                       hessp=hvp,
                       callback=cbk if verbose else None)
        assert res.success
        mu = res.x
        assert np.all(np.isfinite(mu))
        if verbose: print("Done")

        # Unflatten and compute the expected sufficient statistics
        return mu.reshape((T, D))

    def _laplace_approximation_newton(self, tol=1e-6, stepsz=0.9, verbose=False):
        """
        Solve a block tridiagonal system with message passing.
        """
        from pylds.util import solve_symm_block_tridiag, scipy_solve_symm_block_tridiag
        scale = self.T * self.D_emission

        def newton_step(x, stepsz):
            assert 0 <= stepsz <= 1
            g = self.gradient_log_joint(x)
            H_diag, H_upper_diag = self.sparse_hessian_log_joint(x)
            Hinv_g = -scipy_solve_symm_block_tridiag(-H_diag / scale,
                                               -H_upper_diag / scale,
                                               g / scale)
            return x - stepsz * Hinv_g

        if verbose:
            print("Fitting Laplace approximation")

        itr = [0]
        def cbk(x):
            print("Iteration: ", itr[0],
                  "\tObjective: ", (self.log_joint(x) / scale).round(4),
                  "\tAvg Grad: ", (self.gradient_log_joint(x).mean() / scale).round(4))
            itr[0] += 1

        # Solve for optimal x with Newton's method
        x = self.gaussian_states
        dx = np.inf
        while dx >= tol:
            xnew = newton_step(x, stepsz)
            dx = np.mean(abs(xnew - x))
            x = xnew

            if verbose:
                cbk(x)

        assert np.all(np.isfinite(x))
        if verbose:
            print("Done")

        return x

    def log_likelihood(self):
        if self._normalizer is None:
            self.E_step()
        return self._normalizer

    def E_step(self, verbose=False):
        self.gaussian_states = self.laplace_approximation(verbose=verbose)

        # Compute normalizer and covariances with E step
        T, D = self.T, self.D_latent
        H_diag, H_upper_diag = self.sparse_hessian_log_joint(self.gaussian_states)
        J_init = J_11 = J_22 = np.zeros((D, D))
        h_init = h_1 = h_2 = np.zeros((D,))

        # Negate the Hessian since precision is -H
        J_21 = np.swapaxes(-H_upper_diag, -1, -2)
        J_node = -H_diag
        h_node = np.zeros((T, D))

        logZ, _, self.smoothed_sigmas, E_xtp1_xtT = \
            info_E_step(J_init, h_init, 0,
                        J_11, J_21, J_22, h_1, h_2, np.zeros((T - 1)),
                        J_node, h_node, np.zeros(T))

        # Laplace approximation -- normalizer is the joint times
        # the normalizer from the Gaussian approx.
        self._normalizer = self.log_joint(self.gaussian_states) + logZ

        self._set_expected_stats(self.gaussian_states, self.smoothed_sigmas, E_xtp1_xtT)

    def _set_expected_stats(self, mu, sigmas, E_xtp1_xtT):
        # Get the emission stats
        p, n, d, T, inputs, y = \
            self.D_emission, self.D_latent, self.D_input, self.T, \
            self.inputs, self.data

        E_x_xT = sigmas + mu[:, :, None] * mu[:, None, :]
        E_x_uT = mu[:, :, None] * self.inputs[:, None, :]
        E_u_uT = self.inputs[:, :, None] * self.inputs[:, None, :]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT, E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0, 2, 1)), E_u_uT), axis=2)),
            axis=1)
        E_xut_xutT = E_xu_xuT[:-1].sum(0)

        E_xtp1_xtp1T = E_x_xT[1:].sum(0)
        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        E_xtp1_utT = (mu[1:, :, None] * inputs[:-1, None, :]).sum(0)
        E_xtp1_xutT = np.hstack((E_xtp1_xtT, E_xtp1_utT))

        self.E_dynamics_stats = np.array(
            [E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, self.T - 1])

        # Compute the expectations for the observations
        E_yxT = np.sum(y[:, :, None] * mu[:, None, :], axis=0)
        E_yuT = y.T.dot(inputs)
        E_yxuT = np.hstack((E_yxT, E_yuT))
        self.E_emission_stats = np.array([E_yxuT, mu, sigmas, inputs, np.ones_like(y, dtype=bool)])

    def smooth(self):
        return self.emission_distn.predict(np.hstack((self.gaussian_states, self.inputs)))


class LaplaceApproxPoissonLDSStates(_LaplaceApproxLDSStatesBase):
    """
    Poisson observations
    """
    def local_log_likelihood(self, xt, yt, ut):
        # Observation likelihoods
        C, D = self.C, self.D

        loglmbda = np.dot(C, xt) + np.dot(D, ut)
        lmbda = np.exp(loglmbda)

        ll = np.sum(yt * loglmbda)
        ll -= np.sum(lmbda)
        ll -= np.sum(gammaln(yt + 1))
        return ll

    # Override likelihood, gradient, and hessian with vectorized forms
    def log_conditional_likelihood(self, x):
        # Observation likelihoods
        C, D = self.C, self.D

        loglmbda = np.dot(x, C.T) + np.dot(self.inputs, D.T)
        lmbda = np.exp(loglmbda)

        ll = np.sum(self.data * loglmbda)
        ll -= np.sum(lmbda)
        ll -= np.sum(gammaln(self.data + 1))
        return ll

    def grad_local_log_likelihood(self, x):
        """
        d/dx  y^T Cx + y^T d - exp(Cx+d)
            = y^T C - exp(Cx+d)^T C
            = (y - lmbda)^T C
        """
        # Observation likelihoods
        lmbda = np.exp(np.dot(x, self.C.T) + np.dot(self.inputs, self.D.T))
        return (self.data - lmbda).dot(self.C)

    def hessian_local_log_likelihood(self, x):
        """
        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + d)^T C
            = -C^T exp(Cx + d)^T C
        """
        # Observation likelihoods
        lmbda = np.exp(np.dot(x, self.C.T) + np.dot(self.inputs, self.D.T))
        return np.einsum('tn, ni, nj ->tij', -lmbda, self.C, self.C)

    # Test hooks
    def test_joint_probability(self, x):
        # A differentiable function to compute the joint probability for a given
        # latent state sequence
        import autograd.numpy as anp
        T = self.T
        ll = 0

        # Initial likelihood
        mu_init, sigma_init = self.mu_init, self.sigma_init
        ll += -0.5 * anp.dot(x[0] - mu_init, anp.linalg.solve(sigma_init, x[0] - mu_init))

        # Transition likelihoods
        A, B, Q = self.A, self.B, self.sigma_states
        xpred = anp.dot(x[:T-1], A.T) + anp.dot(self.inputs[:T-1], B.T)
        dx = x[1:] - xpred
        ll += -0.5 * (dx.T * anp.linalg.solve(Q, dx.T)).sum()

        # Observation likelihoods
        y = self.data
        C, D = self.C, self.D
        loglmbda = (anp.dot(x, C.T) + anp.dot(self.inputs, D.T))
        lmbda = anp.exp(loglmbda)

        ll += anp.sum(y * loglmbda)
        ll -= anp.sum(lmbda)

        if anp.isnan(ll):
            ll = -anp.inf

        return ll

    def test_gradient_log_joint(self, x):
        return grad(self.test_joint_probability)(x)

    def test_hessian_log_joint(self, x):
        return hessian(self.test_joint_probability)(x)


class LaplaceApproxBernoulliLDSStates(_LaplaceApproxLDSStatesBase):
    """
    Bernoulli observations with Laplace approximation

    Let \psi_t = C x_t + D u_t

    p(y_t = 1 | x_t) = \sigma(\psi_t)

    log p(y_t | x_t) = y_t \log \sigma(\psi_t) +
                       (1-y_t) \log \sigma(-\psi_t) +

                     = y_t \psi_t - log (1 + exp(\psi_t))

    use the log-sum-exp trick to compute this:
                     = y_t \psi_t - log {exp(0) + exp(\psi_t)}
                     = y_t \psi_t - log {m [exp(0 - log m) + exp(\psi_t - log m)]}
                     = y_t \psi_t - log m - log {exp(-log m) + exp(\psi_t - log m)}

    set log m = max(0, psi)

    """
    def local_log_likelihood(self, xt, yt, ut):
        # Observation likelihoods
        C, D = self.C, self.D

        psi = C.dot(xt) + D.dot(ut)

        ll = np.sum(yt * psi)

        # Compute second term with log-sum-exp trick (see above)
        logm = np.maximum(0, psi)
        ll -= np.sum(logm)
        ll -= np.sum(np.log(np.exp(-logm) + np.exp(psi - logm)))
        return ll

    # Override likelihood, gradient, and hessian with vectorized forms
    def log_conditional_likelihood(self, x):
        # Observation likelihoods
        C, D, u, y = self.C, self.D, self.inputs, self.data
        psi = x.dot(C.T) + u.dot(D.T)

        # First term is linear in psi
        ll = np.sum(y * psi)

        # Compute second term with log-sum-exp trick (see above)
        logm = np.maximum(0, psi)
        ll -= np.sum(logm)
        ll -= np.sum(np.log(np.exp(-logm) + np.exp(psi - logm)))

        return ll

    def grad_local_log_likelihood(self, x):
        """
        d/d \psi  y \psi - log (1 + exp(\psi))
            = y - exp(\psi) / (1 + exp(\psi))
            = y - sigma(psi)
            = y - p

        d \psi / dx = C

        d / dx = (y - sigma(psi)) * C
        """
        C, D, u, y = self.C, self.D, self.inputs, self.data
        psi = x.dot(C.T) + u.dot(D.T)
        p = 1. / (1 + np.exp(-psi))
        return (y - p).dot(C)

    def hessian_local_log_likelihood(self, x):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        C, D, u, y = self.C, self.D, self.inputs, self.data
        psi = x.dot(C.T) + u.dot(D.T)
        p = 1. / (1 + np.exp(-psi))
        dp_dpsi = p * (1 - p)
        return np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.C, self.C)

    # Test hooks
    def test_joint_probability(self, x):
        # A differentiable function to compute the joint probability for a given
        # latent state sequence
        import autograd.numpy as anp
        T = self.T
        ll = 0

        # Initial likelihood
        mu_init, sigma_init = self.mu_init, self.sigma_init
        ll += -0.5 * anp.dot(x[0] - mu_init, anp.linalg.solve(sigma_init, x[0] - mu_init))

        # Transition likelihoods
        A, B, Q = self.A, self.B, self.sigma_states
        xpred = anp.dot(x[:T-1], A.T) + anp.dot(self.inputs[:T-1], B.T)
        dx = x[1:] - xpred
        ll += -0.5 * (dx.T * anp.linalg.solve(Q, dx.T)).sum()

        # Observation likelihoods
        y = self.data
        C, D = self.C, self.D
        psi = (anp.dot(x, C.T) + anp.dot(self.inputs, D.T))

        ll += anp.sum(y * psi)
        ll -= anp.sum(np.log(1 + np.exp(psi)))

        if anp.isnan(ll):
            ll = -anp.inf

        return ll

    def test_gradient_log_joint(self, x):
        return grad(self.test_joint_probability)(x)

    def test_hessian_log_joint(self, x):
        return hessian(self.test_joint_probability)(x)
