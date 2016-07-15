from __future__ import division
import numpy as np

from pybasicbayes.util.general import AR_striding, objarray
from pybasicbayes.util.stats import mniw_expectedstats

from lds_messages_interface import kalman_filter, filter_and_sample, E_step, \
    info_E_step, filter_and_sample_diagonal, kalman_filter_diagonal, \
    kalman_info_filter, info_sample


class LDSStates(object):
    def __init__(self,model,T=None,data=None,stateseq=None,
            generate=True,initialize_from_prior=False,initialize_to_noise=True):
        self.model = model
        self.data = data

        self.T = T if T else data.shape[0]
        self.data = data

        self._normalizer = None

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

    ### basics

    def log_likelihood(self):
        if self._normalizer is None:
            self._normalizer, _, _ = kalman_filter(
                self.mu_init, self.sigma_init,
                self.A, self.sigma_states, self.C, self.sigma_obs,
                self.data)
        return self._normalizer

    ### generation

    def generate_states(self):
        T, n = self.T, self.n

        stateseq = self.stateseq = np.empty((T,n),dtype='double')
        stateseq[0] = np.random.multivariate_normal(self.mu_init, self.sigma_init)

        chol = np.linalg.cholesky(self.sigma_states)
        randseq = np.random.randn(T-1,n).dot(chol.T)

        for t in xrange(1,T):
            stateseq[t] = self.A.dot(stateseq[t-1]) + randseq[t-1]

        return stateseq

    def sample_predictions(self, Tpred, states_noise, obs_noise):
        _, filtered_mus, filtered_sigmas = kalman_filter(
            self.mu_init, self.sigma_init, self.A, self.sigma_states, self.C,
            self.sigma_obs, self.data)

        init_mu = self.A.dot(filtered_mus[-1])
        init_sigma = self.sigma_states + self.A.dot(
            filtered_sigmas[-1]).dot(self.A.T)

        randseq = np.zeros((Tpred-1, self.n))
        if states_noise:
            L = np.linalg.cholesky(self.sigma_states)
            randseq += np.random.randn(Tpred-1, self.n).dot(L.T)

        states = np.empty((Tpred, self.n))
        states[0] = np.random.multivariate_normal(init_mu, init_sigma)
        for t in xrange(1,Tpred):
            states[t] = self.A.dot(states[t-1]) + randseq[t-1]

        obs = states.dot(self.C.T)
        if obs_noise:
            L = np.linalg.cholesky(self.sigma_obs)
            obs += np.random.randn(Tpred, self.p).dot(L.T)

        return obs

    ### filtering and smoothing

    def filter(self):
        if self.diagonal_noise:
            self._normalizer, self.filtered_mus, self.filtered_sigmas = \
                kalman_filter_diagonal(
                    self.mu_init, self.sigma_init,
                    self.A, self.sigma_states, self.C, self.sigma_obs_flat,
                    self.data)
        else:
            self._normalizer, self.filtered_mus, self.filtered_sigmas = \
                kalman_filter(
                    self.mu_init, self.sigma_init,
                    self.A, self.sigma_states, self.C, self.sigma_obs,
                    self.data)

    def smooth(self):
        # Use the info E step because it can take advantage of diagonal noise
        # The standard E step could but we have not implemented it
        self.info_E_step()
        return self.smoothed_mus.dot(self.C.T)

    ### resampling

    def resample(self):
        if self.diagonal_noise:
            self._normalizer, self.stateseq = filter_and_sample_diagonal(
                self.mu_init, self.sigma_init,
                self.A, self.sigma_states, self.C, self.sigma_obs_flat,
                self.data)
        else:
            self._normalizer, self.stateseq = filter_and_sample(
                self.mu_init, self.sigma_init,
                self.A, self.sigma_states, self.C, self.sigma_obs,
                self.data)

    ### EM

    def E_step(self):
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = E_step(
                self.mu_init, self.sigma_init,
                self.A, self.sigma_states, self.C, self.sigma_obs,
                self.data)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def _set_expected_stats(self,smoothed_mus,smoothed_sigmas,E_xtp1_xtT):
        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        data = self.data
        EyyT = data.T.dot(data)
        EyxT = data.T.dot(smoothed_mus)
        ExxT = smoothed_sigmas.sum(0) + smoothed_mus.T.dot(smoothed_mus)

        E_xt_xtT = \
            ExxT - (smoothed_sigmas[-1]
                    + np.outer(smoothed_mus[-1],smoothed_mus[-1]))
        E_xtp1_xtp1T = \
            ExxT - (smoothed_sigmas[0]
                    + np.outer(smoothed_mus[0], smoothed_mus[0]))

        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        def is_symmetric(A):
            return np.allclose(A,A.T)

        assert is_symmetric(ExxT)
        assert is_symmetric(E_xt_xtT)
        assert is_symmetric(E_xtp1_xtp1T)

        self.E_emission_stats = np.array([EyyT, EyxT, ExxT, self.T])
        self.E_dynamics_stats = np.array([E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, self.T-1])

    # next two methods are for testing

    def info_E_step(self):
        data = self.data
        A, sigma_states, C, sigma_obs = \
            self.A, self.sigma_states, self.C, self.sigma_obs

        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        J_pair_11 = A.T.dot(np.linalg.solve(sigma_states, A))
        J_pair_21 = -np.linalg.solve(sigma_states, A)
        J_pair_22 = np.linalg.inv(sigma_states)

        # Check if diagonal and avoid inverting D_obs x D_obs matrix
        if self.diagonal_noise:
            J_node = (C.T * 1./ self.sigma_obs_flat).dot(C)
            h_node = np.einsum('ik,i,ti->tk', C, 1./self.sigma_obs_flat, data)
        else:
            J_node = C.T.dot(np.linalg.solve(sigma_obs, C))
            h_node = np.einsum('ik,ij,tj->tk', C, np.linalg.inv(sigma_obs), data)

        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(
                J_init,h_init,J_pair_11,J_pair_21,J_pair_22,J_node,h_node)
        self._normalizer += self._extra_loglike_terms(
            self.A, self.sigma_states, self.C, self.sigma_obs,
            self.mu_init, self.sigma_init, self.data)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    @staticmethod
    def _extra_loglike_terms(A, BBT, C, DDT, mu_init, sigma_init, data):
        p, n = C.shape
        T = data.shape[0]
        out = 0.

        out -= 1./2 * mu_init.dot(np.linalg.solve(sigma_init,mu_init))
        out -= 1./2 * np.linalg.slogdet(sigma_init)[1]
        out -= n/2. * np.log(2*np.pi)

        out -= (T-1)/2. * np.linalg.slogdet(BBT)[1]
        out -= (T-1)*n/2. * np.log(2*np.pi)

        out -= 1./2 * np.einsum('ij,ti,tj->',np.linalg.inv(DDT),data,data)
        out -= T/2. * np.linalg.slogdet(DDT)[1]
        out -= T*p/2 * np.log(2*np.pi)

        return out

    ### mean field

    def meanfieldupdate(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            self.dynamics_distn.meanfield_expectedstats()

        J_yy, J_yx, J_node, logdet_node = \
            self.emission_distn.meanfield_expectedstats()

        h_node = self.data.dot(J_yx)

        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(
                J_init,h_init,J_pair_11,-J_pair_21,J_pair_22,J_node,h_node)

        self._normalizer += self._info_extra_loglike_terms(
            J_init, h_init, logdet_pair, J_yy, logdet_node, self.data)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def get_vlb(self):
        if not hasattr(self,'_normalizer'):
            self.meanfieldupdate()  # NOTE: sets self._normalizer
        return self._normalizer

    @staticmethod
    def _info_extra_loglike_terms(
            J_init, h_init, logdet_pair, J_yy, logdet_node, data):
        p, n, T = J_yy.shape[0], h_init.shape[0], data.shape[0]

        out = 0.

        out -= 1./2 * h_init.dot(np.linalg.solve(J_init, h_init))
        out += 1./2 * np.linalg.slogdet(J_init)[1]
        out -= n/2. * np.log(2*np.pi)

        out += 1./2 * logdet_pair.sum() if isinstance(logdet_pair, np.ndarray) \
            else (T-1)/2. * logdet_pair
        out -= (T-1)*n/2. * np.log(2*np.pi)

        contract = 'ij,ti,tj->' if J_yy.ndim == 2 else 'tij,ti,tj->'
        out -= 1./2 * np.einsum(contract, J_yy, data, data)
        out += 1./2 * logdet_node.sum() if isinstance(logdet_node, np.ndarray) \
            else T/2. * logdet_node
        out -= T*p/2. * np.log(2*np.pi)

        return out

    # model properties

    @property
    def emission_distn(self):
        return self.model.emission_distn

    @property
    def diagonal_noise(self):
        return self.model.diagonal_noise

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
    def sigma_obs_flat(self):
        return self.model.sigma_obs_flat

    @property
    def strided_stateseq(self):
        return AR_striding(self.stateseq,1)


class LDSStatesMissingData(LDSStates):
    """
    Some regression models can also handle missing observations. For example,
    the DiagonalRegression class allows the data to be given along with a binary
    mask that indicates whether the entry is present. In order to support that
    here, we need to:
        1. Update the constructor to take a mask
        2. Update likelihood calculation to use info form, where Jnode (obs precision)
           equals zero for missing observations.
        3. Update filter and smooth to use info form.
        4. Update the sampling code to use info form.
        5. Update meanfield code to use info form.
        6. Update the E_emission_stats to include pixel-wise latent state covariances

    Since DiagonalRegression is the only emission model that supports
    """

    def __init__(self, model, mask=None, **kwargs):
        super(LDSStatesMissingData, self).__init__(model, **kwargs)

        self.mask = mask if mask is not None else np.ones_like(self.data, dtype=bool)

    @property
    def info_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        J_pair_11 = self.A.T.dot(np.linalg.solve(self.sigma_states, self.A))
        J_pair_21 = -np.linalg.solve(self.sigma_states, self.A)
        J_pair_22 = np.linalg.inv(self.sigma_states)

        if self.diagonal_noise:
            J_node_slow = np.array([(self.C.T * self.mask[t] / self.sigma_obs_flat).dot(self.C) for t in range(self.T)])
            J_node = np.einsum('ji,tj,jk->tik', self.C, self.mask / self.sigma_obs_flat, self.C)
            assert np.allclose(J_node, J_node_slow)
            h_node_slow = np.einsum('ik,ti,ti->tk', self.C, self.mask / self.sigma_obs_flat, self.data)
            h_node = (self.data * self.mask / self.sigma_obs_flat).dot(self.C)
            assert np.allclose(h_node, h_node_slow)
        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

        return J_init, h_init, J_pair_11, J_pair_21, J_pair_22, \
               J_node, h_node

    @property
    def expected_info_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            self.dynamics_distn.meanfield_expectedstats()

        # Negate J_pair_21 (np.linalg.solve(sigma_states, A)
        J_pair_21 = -J_pair_21

        if self.diagonal_noise:
            # Use the fact that the diagonalregression prior is factorized
            E_W, E_WWT, E_sigmasq_inv, _ = self.emission_distn.mf_expectations
            E_WWT_vec = E_WWT.reshape(self.p, -1)
            Jobs = self.mask * E_sigmasq_inv
            J_node = (np.dot(Jobs, E_WWT_vec)).reshape((self.T, self.n, self.n))
            h_node = (self.data * Jobs).dot(E_W)

        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

        return J_init, h_init, J_pair_11, J_pair_21, J_pair_22, J_node, h_node

    def log_likelihood(self):
        if self._normalizer is None:
            self._normalizer, _, _ = kalman_info_filter(*self.info_params)
        return self._normalizer

    def filter(self):
        if self.diagonal_noise:
            self._normalizer, self.filtered_mus, self.filtered_sigmas = \
                kalman_info_filter(*self.info_params)
                # kalman_filter_diagonal(
                #     self.mu_init, self.sigma_init,
                #     self.A, self.sigma_states, self.C, self.sigma_obs_flat,
                #     self.data)
        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")


    def smooth(self):
        # Use the info E step because it can take advantage of diagonal noise
        # The standard E step could but we have not implemented it
        self.info_E_step()
        return self.smoothed_mus.dot(self.C.T)

    def info_E_step(self):
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
        E_xtp1_xtT = info_E_step(*self.info_params)
        self._normalizer += self._extra_loglike_terms(
            self.A, self.sigma_states, self.C, self.sigma_obs,
            self.mu_init, self.sigma_init, self.data)

        self._set_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def resample(self):
        if self.diagonal_noise:
            self._normalizer, self.stateseq = info_sample(*self.info_params)
        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

    def E_step(self):
        return self.info_E_step()

    def meanfieldupdate(self):
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(*self.expected_info_params)

        # TODO: Update the normalization code
        # self._normalizer += self._info_extra_loglike_terms(
        #     J_init, h_init, logdet_pair, J_yy, logdet_node, self.data)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)


    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        p, n, T, data, mask = self.p, self.n, self.T, self.data, self.mask
        ExxT = smoothed_sigmas.sum(0) + smoothed_mus.T.dot(smoothed_mus)

        E_xt_xtT = \
            ExxT - (smoothed_sigmas[-1]
                    + np.outer(smoothed_mus[-1], smoothed_mus[-1]))
        E_xtp1_xtp1T = \
            ExxT - (smoothed_sigmas[0]
                    + np.outer(smoothed_mus[0], smoothed_mus[0]))

        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        def is_symmetric(A):
            return np.allclose(A, A.T)

        assert is_symmetric(ExxT)
        assert is_symmetric(E_xt_xtT)
        assert is_symmetric(E_xtp1_xtp1T)

        self.E_dynamics_stats = np.array([E_xtp1_xtp1T, E_xtp1_xtT, E_xt_xtT, self.T - 1])

        # Get the emission stats
        E_ysq = np.sum(data**2 * mask, axis=0)
        E_yxT = (data * mask).T.dot(smoothed_mus)
        E_xxT_vec = smoothed_sigmas.reshape((T, n**2))
        E_xxT = np.array([np.dot(self.mask[:, d], E_xxT_vec).reshape((n, n)) for d in range(p)])
        Tp = np.sum(self.mask, axis=0)

        self.E_emission_stats = objarray([E_ysq, E_yxT, E_xxT, Tp])
