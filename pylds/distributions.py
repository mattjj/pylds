import autograd.numpy as np
from autograd import value_and_grad, hessian_vector_product
from autograd.scipy.special import gammaln

from scipy.optimize import minimize

from pybasicbayes.distributions import Regression

class PoissonRegression(Regression):
    """
    Poisson regression with Gaussian distributed inputs and exp link:

       y ~ Poisson(exp(Ax + b))

    where x ~ N(mu, sigma)

    Currently, we only support maximum likelihood estimation of the
    parameters, A and b, given the distribution over inputs, x, and
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

    def log_likelihood(self,xy):
        assert isinstance(xy, tuple)
        x, y = xy
        loglmbda = x.dot(self.A.T) + self.b.T
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

    @property
    def D_in(self):
        return self._D_in

    @property
    def D_out(self):
        return self._D_out

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
        EyxT = np.sum([s[0] for s in stats], axis=0)
        mus = np.vstack([s[1] for s in stats])
        sigs = np.vstack([s[2] for s in stats])
        masks = np.vstack(s[3] for s in stats)
        T = mus.shape[0]
        D = self.D_in

        # Optimize each row of A independently
        for n in range(self.D_out):

            # Flatten the covariance to enable vectorized calculations
            sigs_vec = sigs.reshape((T,D**2))
            def ll_vec(an):

                ll = 0
                ll += np.dot(an, EyxT[n])

                # Vectorized log likelihood calculation
                loglmbda = np.dot(mus, an)
                aa_vec = np.outer(an, an).reshape((D ** 2,))
                trms = np.exp(loglmbda + 0.5 * np.dot(sigs_vec, aa_vec))
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
            # res = minimize(value_and_grad(obj), self.A[n],
            #                tol=1e-3,
            #                method="Newton-CG",
            #                jac=True,
            #                hessp=hessian_vector_product(obj),
            #                callback=cbk if verbose else None)
            assert res.success
            self.A[n] = res.x
