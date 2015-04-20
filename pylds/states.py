from __future__ import division
import numpy as np

from pybasicbayes.util.general import AR_striding

from lds_messages_interface import kalman_filter, filter_and_sample, E_step, \
    kalman_info_filter

class LDSStates(object):
    def __init__(self,model,T=None,data=None,stateseq=None,
            generate=True,initialize_from_prior=False,initialize_to_noise=True):
        self.model = model
        self.data = data

        self.T = T if T else data.shape[0]
        self.data = data

        if stateseq is not None:
            self.stateseq = stateseq
        elif generate:
            if data is not None and not (initialize_from_prior or initialize_to_noise):
                self.resample()
            else:
                if initialize_from_prior:
                    self.generate_states()
                else:
                    self.stateseq = np.random.normal(size=(self.T,self.n))

    ### model properties

    @property
    def emission_distn(self):
        return self.model.emission_distn

    @property
    def dynamics_distn(self):
        return self.model.dynamics_distn

    @property
    def mu_init(self):
        return self.model.mu_init

    @property
    def sigma_init(self):
        return self.model.sigma_init

    @property
    def n(self):
        return self.model.n

    @property
    def p(self):
        return self.model.p

    @property
    def A(self):
        return self.model.A

    @property
    def sigma_states(self):
        return self.model.sigma_states

    @property
    def C(self):
        return self.model.C

    @property
    def sigma_obs(self):
        return self.model.sigma_obs

    @property
    def strided_stateseq(self):
        return AR_striding(self.stateseq,1)

    def log_likelihood(self):
        # TODO handle caching and stuff
        if True or self._normalizer is None:
            self._normalizer, _, _ = kalman_filter(
                self.mu_init, self.sigma_init,
                self.A, self.sigma_states, self.C, self.sigma_obs,
                self.data)
        return self._normalizer

    # generation

    def generate_states(self):
        T, n, p = self.T, self.n, self.p

        stateseq = self.stateseq = np.empty((T,n),dtype='double')
        stateseq[0] = np.random.multivariate_normal(self.mu_init, self.sigma_init)

        chol = np.linalg.cholesky(self.sigma_states)
        randseq = np.random.randn(T-1,n)

        for t in xrange(1,T):
            stateseq[t] = self.A.dot(stateseq[t-1]) + chol.dot(randseq[t-1])

        return stateseq

    # filtering

    def filter(self):
        self._normalizer, self.filtered_mus, self.filtered_sigmas = kalman_filter(
            self.mu_init, self.sigma_init,
            self.A, self.sigma_states, self.C, self.sigma_obs,
            self.data)

    # resampling

    def resample(self):
        self._normalizer, self.stateseq = filter_and_sample(
            self.mu_init, self.sigma_init,
            self.A, self.sigma_states, self.C, self.sigma_obs,
            self.data)

    # E step

    def E_step(self):
        assert np.allclose(self.sigma_states, self.sigma_states.T)
        assert np.allclose(self.sigma_obs, self.sigma_obs.T)
        assert np.all(np.linalg.eigvalsh(self.sigma_states) > 0)
        assert np.all(np.linalg.eigvalsh(self.sigma_obs) > 0)

        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtTs = E_step(
                self.mu_init, self.sigma_init,
                self.A, self.sigma_states, self.C, self.sigma_obs,
                self.data)

        assert not np.isnan(E_xtp1_xtTs).any()
        assert not np.isnan(self.smoothed_mus).any()
        assert not np.isnan(self.smoothed_sigmas).any()
        assert not np.isnan(self._normalizer)

        # TODO maybe put these in the low-level code too...
        EyyT = np.einsum('ti,tj->ij',self.data,self.data)
        EyxT = np.einsum('ti,tj->ij',self.data,self.smoothed_mus)
        ExxT = self.smoothed_sigmas.sum(0) + \
            np.einsum('ti,tj->ij',self.smoothed_mus,self.smoothed_mus)

        E_xt_xtT = \
            ExxT - (self.smoothed_sigmas[-1]
                    + np.outer(self.smoothed_mus[-1],self.smoothed_mus[-1]))
        E_xtp1_xtp1T = \
            ExxT - (self.smoothed_sigmas[0]
                    + np.outer(self.smoothed_mus[0], self.smoothed_mus[0]))

        E_xtp1_xtT = E_xtp1_xtTs.sum(0)

        def is_symmetric(A):
            return np.allclose(A,A.T)
        assert is_symmetric(ExxT)
        assert is_symmetric(E_xt_xtT)
        assert is_symmetric(E_xtp1_xtp1T)

        self.E_emission_stats = np.array([EyyT, EyxT, ExxT, self.T])
        self.E_dynamics_stats = \
            np.array([E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, self.T-1])

