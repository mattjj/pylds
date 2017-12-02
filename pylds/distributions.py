import autograd.numpy as np
from autograd import value_and_grad
from autograd.scipy.special import gammaln

from scipy.optimize import minimize

from pybasicbayes.distributions import Regression
from pybasicbayes.util.text import progprint_xrange


class PoissonRegression(Regression):
    """
    Poisson regression with Gaussian distributed inputs and exp link:

       y ~ Poisson(exp(Ax))

    where x ~ N(mu, sigma)

    Currently, we only support maximum likelihood estimation of the
    parameters A given the distribution over inputs, x, and
    the observed outputs, y.

    We compute the expected log likelihood in closed form (since
    we can do this with the exp link function), and we use Autograd
    to compute its gradients.
    """

    def __init__(self, D_out, D_in, A=None, verbose=False):
        self._D_out, self._D_in = D_out, D_in
        self.verbose = verbose

        if A is not None:
            assert A.shape == (D_out, D_in)
            self.A = A.copy()
        else:
            self.A = 0.01 * np.random.randn(D_out, D_in)

        self.sigma = None

    @property
    def D_in(self):
        return self._D_in

    @property
    def D_out(self):
        return self._D_out

    def log_likelihood(self,xy):
        assert isinstance(xy, tuple)
        x, y = xy
        loglmbda = x.dot(self.A.T)
        lmbda = np.exp(loglmbda)
        return -gammaln(y+1) - lmbda + y * loglmbda

    def expected_log_likelihood(self, mus, sigmas, y):
        """
        Compute the expected log likelihood for a mean and
        covariance of x and an observed value of y.
        """

        # Flatten the covariance
        T = mus.shape[0]
        D = self.D_in
        sigs_vec = sigmas.reshape((T, D ** 2))

        # Compute the log likelihood of each column
        ll = np.zeros((T, self.D_out))
        for n in range(self.D_out):

            an = self.A[n]

            E_loglmbda = np.dot(mus, an)
            ll[:,n] += y[:,n] * E_loglmbda

            # Vectorized log likelihood calculation
            aa_vec = np.outer(an, an).reshape((D ** 2,))
            ll[:,n] = -np.exp(E_loglmbda + 0.5 * np.dot(sigs_vec, aa_vec))

        return ll

    def predict(self, x):
        return np.exp(x.dot(self.A.T))

    def rvs(self,x=None,size=1,return_xy=True):
        x = np.random.normal(size=(size, self.D_in)) if x is None else x
        y = np.random.poisson(self.predict(x))
        return np.hstack((x, y)) if return_xy else y

    def max_likelihood(self, data, weights=None,stats=None):
        """
        Maximize the likelihood for a given value of x
        :param data:
        :param weights:
        :param stats:
        :return:
        """
        raise NotImplementedError

    def max_expected_likelihood(self, stats, verbose=False):
        # These aren't really "sufficient" statistics, since we
        # need the mean and covariance for each time bin.
        EyxuT = np.sum([s[0] for s in stats], axis=0)
        mus = np.vstack([s[1] for s in stats])
        sigmas = np.vstack([s[2] for s in stats])
        inputs = np.vstack([s[3] for s in stats])
        masks = np.vstack(s[4] for s in stats)
        T = mus.shape[0]

        D_latent = mus.shape[1]
        sigmas_vec = sigmas.reshape((T, D_latent**2))

        # Optimize each row of A independently
        ns = progprint_xrange(self.D_out) if verbose else range(self.D_out)
        for n in ns:

            # Flatten the covariance to enable vectorized calculations
            def ll_vec(an):

                ll = 0
                ll += np.dot(an, EyxuT[n])

                # Vectorized log likelihood calculation
                loglmbda = np.dot(mus, an)
                aa_vec = np.outer(an[:D_latent], an[:D_latent]).reshape((D_latent ** 2,))
                trms = np.exp(loglmbda + 0.5 * np.dot(sigmas_vec, aa_vec))
                ll -= np.sum(trms[masks[:, n]])

                if not np.isfinite(ll):
                    return -np.inf

                return ll / T
            obj = lambda x: -ll_vec(x)

            itr = [0]
            def cbk(x):
                itr[0] += 1
                print("M_step iteration ", itr[0])

            res = minimize(value_and_grad(obj), self.A[n],
                           jac=True,
                           callback=cbk if verbose else None)
            assert res.success
            self.A[n] = res.x


class BernoulliRegression(Regression):
    """
    Bernoulli regression with Gaussian distributed inputs and logistic link:

       y ~ Bernoulli(logistic(Ax))

    where x ~ N(mu, sigma)

    Currently, we only support maximum likelihood estimation of the
    parameter A given the distribution over inputs, x, and
    the observed outputs, y.

    We approximate the expected log likelihood with Monte Carlo.
    """

    def __init__(self, D_out, D_in, A=None, verbose=False):
        self._D_out, self._D_in = D_out, D_in
        self.verbose = verbose

        if A is not None:
            assert A.shape == (D_out, D_in)
            self.A = A.copy()
        else:
            self.A = 0.01 * np.random.randn(D_out, D_in)

        self.sigma = None

    @property
    def D_in(self):
        return self._D_in

    @property
    def D_out(self):
        return self._D_out

    def log_likelihood(self,xy):
        assert isinstance(xy, tuple)
        x, y = xy
        psi = x.dot(self.A.T)

        # First term is linear
        ll = y * psi

        # Compute second term with log-sum-exp trick (see above)
        logm = np.maximum(0, psi)
        ll -= np.sum(logm)
        ll -= np.sum(np.log(np.exp(-logm) + np.exp(psi - logm)))

        return ll

    def predict(self, x):
        return 1 / (1 + np.exp(-x.dot(self.A.T)))

    def rvs(self, x=None, size=1, return_xy=True):
        x = np.random.normal(size=(size, self.D_in)) if x is None else x
        y = np.random.rand(x.shape[0], self.D_out) < self.predict(x)
        return np.hstack((x, y)) if return_xy else y

    def max_likelihood(self, data, weights=None, stats=None):
        """
        Maximize the likelihood for given data
        :param data:
        :param weights:
        :param stats:
        :return:
        """
        if isinstance(data, list):
            x = np.vstack([d[0] for d in data])
            y = np.vstack([d[1] for d in data])
        elif isinstance(data, tuple):
            assert len(data) == 2
        elif isinstance(data, np.ndarray):
            x, y = data[:,:self.D_in], data[:, self.D_in:]
        else:
            raise Exception("Invalid data type")

        from sklearn.linear_model import LogisticRegression
        for n in progprint_xrange(self.D_out):
            lr = LogisticRegression(fit_intercept=False)
            lr.fit(x, y[:,n])
            self.A[n] = lr.coef_


    def max_expected_likelihood(self, stats, verbose=False, n_smpls=1):

        # These aren't really "sufficient" statistics, since we
        # need the mean and covariance for each time bin.
        EyxuT = np.sum([s[0] for s in stats], axis=0)
        mus = np.vstack([s[1] for s in stats])
        sigmas = np.vstack([s[2] for s in stats])
        inputs = np.vstack([s[3] for s in stats])
        T = mus.shape[0]

        D_latent = mus.shape[1]

        # Draw Monte Carlo samples of x
        sigmas_chol = np.linalg.cholesky(sigmas)
        x_smpls = mus[:, :, None] + np.matmul(sigmas_chol, np.random.randn(T, D_latent, n_smpls))

        # Optimize each row of A independently
        ns = progprint_xrange(self.D_out) if verbose else range(self.D_out)
        for n in ns:

            def ll_vec(an):
                ll = 0

                # todo include mask
                # First term is linear in psi
                ll += np.dot(an, EyxuT[n])

                # Second term depends only on x and cannot be computed in closed form
                # Instead, Monte Carlo sample x
                psi_smpls = np.einsum('tdm, d -> tm', x_smpls, an[:D_latent])
                psi_smpls = psi_smpls + np.dot(inputs, an[D_latent:])[:, None]
                logm = np.maximum(0, psi_smpls)
                trm2_smpls = logm + np.log(np.exp(-logm) + np.exp(psi_smpls - logm))
                ll -= np.sum(trm2_smpls) / n_smpls

                if not np.isfinite(ll):
                    return -np.inf

                return ll / T

            obj = lambda x: -ll_vec(x)

            itr = [0]
            def cbk(x):
                itr[0] += 1
                print("M_step iteration ", itr[0])

            res = minimize(value_and_grad(obj), self.A[n],
                           jac=True,
                           # callback=cbk if verbose else None)
                           callback=None)
            assert res.success
            self.A[n] = res.x

