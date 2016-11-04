from __future__ import division
import numpy as np

from pybasicbayes.util.general import AR_striding, objarray
from pybasicbayes.util.stats import mniw_expectedstats

from pylds.lds_messages_interface import kalman_filter, filter_and_sample, E_step, \
    info_E_step, filter_and_sample_diagonal, kalman_filter_diagonal, \
    kalman_info_filter, info_sample


class LDSStates(object):
    def __init__(self,model,T=None,data=None,inputs=None,stateseq=None,
            generate=True,initialize_from_prior=False,initialize_to_noise=True):
        self.model = model
        self.data = data

        self.T = T if T else data.shape[0]
        self.data = data
        self.inputs = np.zeros((self.T,0)) if inputs is None else inputs

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
                self.A, self.B, self.sigma_states,
                self.C, self.D, self.sigma_obs,
                self.inputs, self.data)
        return self._normalizer

    ### generation

    def generate_states(self):
        T, n = self.T, self.n

        stateseq = self.stateseq = np.empty((T,n),dtype='double')
        stateseq[0] = np.random.multivariate_normal(self.mu_init, self.sigma_init)

        chol = np.linalg.cholesky(self.sigma_states)
        randseq = np.random.randn(T-1,n).dot(chol.T)

        for t in range(1,T):
            stateseq[t] = self.A.dot(stateseq[t-1]) + \
                          self.B.dot(self.inputs[t-1]) + \
                          randseq[t-1]

        return stateseq

    def sample_predictions(self, Tpred, inputs=None, states_noise=False, obs_noise=False):
        inputs = np.zeros((Tpred, self.d)) if inputs is None else inputs
        _, filtered_mus, filtered_sigmas = kalman_filter(
            self.mu_init, self.sigma_init,
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.inputs, self.data)

        init_mu = self.A.dot(filtered_mus[-1]) + self.B.dot(inputs[-1])
        init_sigma = self.sigma_states + self.A.dot(
            filtered_sigmas[-1]).dot(self.A.T)

        randseq = np.zeros((Tpred-1, self.n))
        if states_noise:
            L = np.linalg.cholesky(self.sigma_states)
            randseq += np.random.randn(Tpred-1, self.n).dot(L.T)

        states = np.empty((Tpred, self.n))
        states[0] = np.random.multivariate_normal(init_mu, init_sigma)
        for t in range(1,Tpred):
            states[t] = self.A.dot(states[t-1]) + \
                        self.B.dot(inputs[t-1]) + \
                        randseq[t-1]

        obs = states.dot(self.C.T) + inputs.dot(self.D.T)
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
        return self.smoothed_mus.dot(self.C.T) + self.inputs.dot(self.D.T)

    ### resampling

    def resample(self):
        if self.diagonal_noise:
            self._normalizer, self.stateseq = filter_and_sample_diagonal(
                self.mu_init, self.sigma_init,
                self.A, self.B, self.sigma_states,
                self.C, self.D, self.sigma_obs_flat,
                self.inputs, self.data)
        else:
            self._normalizer, self.stateseq = filter_and_sample(
                self.mu_init, self.sigma_init,
                self.A, self.B, self.sigma_states,
                self.C, self.D, self.sigma_obs,
                self.inputs, self.data)

    ### EM

    def E_step(self):
        # TODO: Update normalizer?
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = E_step(
                self.mu_init, self.sigma_init,
                self.A, self.B, self.sigma_states,
                self.C, self.D, self.sigma_obs,
                self.inputs, self.data)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        # Get the emission stats
        p, n, d, T, inputs, data = self.p, self.n, self.d, self.T, self.inputs, self.data
        E_x_xT = smoothed_sigmas + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]
        E_x_uT = smoothed_mus[:,:,None] * self.inputs[:,None,:]
        E_u_uT = self.inputs[:,:,None] * self.inputs[:,None,:]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT, E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0,2,1)), E_u_uT), axis=2)),
            axis=1)
        E_xut_xutT = E_xu_xuT[:-1].sum(0)

        E_xtp1_xtp1T = E_x_xT[1:].sum(0)
        E_xt_xtT = E_x_xT[:-1].sum(0)
        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        E_xtp1_utT = (smoothed_mus[1:,:,None] * inputs[:-1, None, :]).sum(0)
        E_xtp1_xutT = np.hstack((E_xtp1_xtT, E_xtp1_utT))


        def is_symmetric(A):
            return np.allclose(A, A.T)

        assert is_symmetric(E_xt_xtT)
        assert is_symmetric(E_xtp1_xtp1T)

        self.E_dynamics_stats = np.array(
            [E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, self.T - 1])

        # Emission statistics
        E_yyT = data.T.dot(data)
        E_yxT = data.T.dot(smoothed_mus)
        E_yuT = data.T.dot(inputs)
        E_yxuT = np.hstack((E_yxT, E_yuT))

        self.E_emission_stats = objarray([E_yyT, E_yxuT, E_xu_xuT.sum(0), T])

    # next two methods are for testing

    def info_E_step(self):
        """
        Info form (natural parameterization of the Gaussian) expectations.
        Compute marginal E[x_t | y_{1:T}] = N(x_t | J_t, h_t)
        where J_t = Sigma_t^{-1}
              h_t = Sigma_t^{-1} mu_t

        Write p(x_{t+1} | x_t) = psi(x_{t+1}, x_t)
            = exp(-1/2 [x_t, x_{t+1}]^T [J_11 J_12] [x_t    ]
                                        [J_21 J_22] [x_{t+1}]

                    + [h_1, h_2]^T [x_t    ]
                                   [x_{t+1}]

                    + log Z)

        In an LDS in standard form (Q = dynamics noise)
            J_11 = A^T Q^{-1} A
            J_12 = -A^T Q^{-1}
            J_21 = J_12^T
            J_22 = Q^{-1}

            h_1 = - u^T B^T Q^{-1} A
            h_2 =   u^T B^T Q^{-1}

        Combined with the following message passing prior,
            p(x_t) = exp(-1/2 x_t^T J_t x_t + h_t^T x_t - log Z_t),

        we obtain a joint distribution with info form parameters:

        Jjoint = [J_11 + J_t,     J_12]
                 [J_21            J_22]

        hjoint = [h_1 + h_t,      h_2]

        Furthermore, assume the observation potentials are Gaussian:

        psi(x_t, y_t)
            = exp(-1/2 x_t^T J_node x_t + h_node^T x_t + log Z_node)

        where, with R = observation noise,
            J_node = C^T R^{-1} C
            h_node = (y - Du)^T R^{-1} C

        """
        inputs, data = self.inputs, self.data
        A, B, sigma_states = self.A, self.B, self.sigma_states
        C, D, sigma_obs = self.C, self.D, self.sigma_obs

        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        J_pair_11 = A.T.dot(np.linalg.solve(sigma_states, A))
        J_pair_21 = -np.linalg.solve(sigma_states, A)
        J_pair_22 = np.linalg.inv(sigma_states)

        h_pair_1 = inputs.dot(B.T).dot(J_pair_21)
        h_pair_2 = inputs.dot(np.linalg.solve(sigma_states, B).T)

        # Check if diagonal and avoid inverting D_obs x D_obs matrix
        if self.diagonal_noise:
            J_node = (C.T * 1./ self.sigma_obs_flat).dot(C)
            h_node = np.einsum('ik,i,ti->tk', C, 1./self.sigma_obs_flat, data)
        else:
            J_node = C.T.dot(np.linalg.solve(sigma_obs, C))
            h_node = np.einsum('ik,ij,tj->tk', C, np.linalg.inv(sigma_obs), data)

        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(
                J_init, h_init,
                J_pair_11, J_pair_21, J_pair_22,
                h_pair_1, h_pair_2,
                J_node, h_node)

        self._normalizer += self._extra_loglike_terms(
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.mu_init, self.sigma_init,
            self.inputs, self.data, isdiag=self.diagonal_noise)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    @staticmethod
    def _extra_loglike_terms(A, B, sigma_states,
                             C, D, sigma_obs,
                             mu_init, sigma_init,
                             inputs, data, isdiag=False):
        # TODO: Update with h_pair_1 and h_pair_2
        p, n = C.shape
        T = data.shape[0]
        out = 0.

        # Initial distribution
        out -= 1./2 * mu_init.dot(np.linalg.solve(sigma_init,mu_init))
        out -= 1./2 * np.linalg.slogdet(sigma_init)[1]
        out -= n/2. * np.log(2*np.pi)

        # Dynamics distribution
        out -= (T-1)/2. * np.linalg.slogdet(sigma_states)[1]
        out -= (T-1)*n/2. * np.log(2*np.pi)
        out -= 1./2 * np.einsum('ij,ti,tj->', B.T.dot(np.linalg.solve(sigma_states, B)), inputs, inputs)

        # Observation distribution
        if isdiag:
            out -= 1. / 2 * np.sum(data ** 2 / np.diag(sigma_obs))
            # TODO: Finish me!
        else:
            dt = inputs.dot(D.T)  # Check diagonal
            sigma_obs_inv = np.linalg.inv(sigma_obs)
            out -= 1. / 2 * np.einsum('ij,ti,tj->', sigma_obs_inv, data, data)
            out += np.einsum('ij,ti,tj->', sigma_obs_inv, data, dt)
            out -= 1./2 * np.einsum('ij,ti,tj->', sigma_obs_inv, dt, dt)
            out -= T/2. * np.linalg.slogdet(sigma_obs)[1]
        out -= T*p/2 * np.log(2*np.pi)

        return out

    ### mean field

    def meanfieldupdate(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            self.dynamics_distn.meanfield_expectedstats()

        # TODO: Check the logic behind these expectations.
        # Do we need to worry about correlations between A,B in E_BT_Qinv_A, for example?
        E_Qinv = J_pair_22
        E_AT_Qinv = J_pair_21[:,:self.n].T.copy("C")
        E_BT_Qinv = J_pair_21[:,self.n:].T
        E_BT_Qinv_A = E_BT_Qinv.dot(np.linalg.solve(E_Qinv, E_AT_Qinv.T))
        E_AT_Qinv_A = J_pair_11[:self.n, :self.n].copy("C")

        h_pair_1 = -self.inputs.dot(E_BT_Qinv_A)
        h_pair_2 = self.inputs.dot(E_BT_Qinv)

        J_yy, J_yx, J_node, logdet_node = \
            self.emission_distn.meanfield_expectedstats()
        E_Rinv = J_yy
        E_Rinv_C = J_yx[:,:self.n].copy("C")
        E_Rinv_D = J_yx[:,self.n:].copy("C")
        E_DT_Rinv_C = E_Rinv_D.T.dot(np.linalg.solve(E_Rinv, E_Rinv_C))
        E_CT_Rinv_C = J_node[:self.n, :self.n].copy("C")

        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        h_node = self.data.dot(E_Rinv_C) - self.inputs.dot(E_DT_Rinv_C)

        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(
                J_init,h_init,E_AT_Qinv_A,-E_AT_Qinv, E_Qinv, h_pair_1,h_pair_2,E_CT_Rinv_C,h_node)
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
            J_init, h_init, logdet_pair, J_yy, logdet_node, data,
            isdiag=False):
        # TODO: Fix me!
        p, n, T = J_yy.shape[0], h_init.shape[0], data.shape[0]

        out = 0.

        # Initial distribution
        out -= 1./2 * h_init.dot(np.linalg.solve(J_init, h_init))
        out += 1./2 * np.linalg.slogdet(J_init)[1]
        out -= n/2. * np.log(2*np.pi)

        # dynamics distribution
        out += 1./2 * logdet_pair.sum() if isinstance(logdet_pair, np.ndarray) \
            else (T-1)/2. * logdet_pair
        out -= (T-1)*n/2. * np.log(2*np.pi)

        if isdiag:
            assert (J_yy.ndim == 1 and J_yy.shape[0] == data.shape[1]) or \
                   (J_yy.shape == data.shape)
            out -= 1./2 * np.sum(data**2 * J_yy)
        else:
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
    def d(self):
        return self.model.d

    @property
    def A(self):
        return self.model.A

    @property
    def B(self):
        return self.model.B

    @property
    def sigma_states(self):
        return self.model.sigma_states

    @property
    def C(self):
        return self.model.C

    @property
    def D(self):
        return self.model.D

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

        # Check if diagonal and avoid inverting D_obs x D_obs matrix
        h_pair_1 = self.inputs.dot(self.B.T).dot(J_pair_21)
        h_pair_2 = self.inputs.dot(np.linalg.solve(self.sigma_states, self.B).T)

        if self.diagonal_noise:
            Jobs = self.mask / self.sigma_obs_flat
            CCT = np.array([np.outer(cp,cp) for cp in self.C])
            CCT_vec = np.reshape(CCT, (self.p,self.n**2))
            J_node = (np.dot(Jobs, CCT_vec)).reshape((self.T, self.n, self.n))
            h_node = (self.data * Jobs).dot(self.C)

            # TODO Subtract inputs!
        else:
            raise NotImplementedError("Only supporting diagonal observations for missing data")

        return J_init, h_init, J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, J_node, h_node

    @property
    def expected_info_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            self.dynamics_distn.meanfield_expectedstats()

        # TODO: Check the logic behind these expectations.
        # Do we need to worry about correlations between A,B in E_BT_Qinv_A, for example?
        E_Qinv = J_pair_22
        E_AT_Qinv = J_pair_21[:, :self.n].T.copy("C")
        E_BT_Qinv = J_pair_21[:, self.n:].T
        E_BT_Qinv_A = E_BT_Qinv.dot(np.linalg.solve(E_Qinv, E_AT_Qinv.T))
        E_AT_Qinv_A = J_pair_11[:self.n, :self.n].copy("C")

        h_pair_1 = -self.inputs.dot(E_BT_Qinv_A)
        h_pair_2 = self.inputs.dot(E_BT_Qinv)

        # J_yy, J_yx, J_node, logdet_node = \
        #     self.emission_distn.meanfield_expectedstats()
        # E_Rinv = J_yy
        # E_Rinv_C = J_yx[:, :self.n].copy("C")
        # E_Rinv_D = J_yx[:, self.n:].copy("C")
        # E_DT_Rinv_C = E_Rinv_D.T.dot(np.linalg.solve(E_Rinv, E_Rinv_C))
        # E_CT_Rinv_C = J_node[:self.n, :self.n].copy("C")

        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        # h_node = self.data.dot(E_Rinv_C) - self.inputs.dot(E_DT_Rinv_C)

        if self.diagonal_noise:
            # Use the fact that the diagonalregression prior is factorized
            E_C, E_CCT, E_sigmasq_inv, _ = self.emission_distn.mf_expectations
            E_C, E_D = E_C[:, :self.n], E_C[:, self.n:]
            E_CCT_vec = E_CCT.reshape(self.p, -1)

            J_obs = self.mask * E_sigmasq_inv
            J_node = (np.dot(J_obs, E_CCT_vec)).reshape((self.T, self.n, self.n))
            h_node = (self.data * J_obs).dot(E_C) - (self.inputs.dot(E_D.T) * J_obs).dot(E_C)

        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

        return J_init, h_init, E_AT_Qinv_A, -E_AT_Qinv, E_Qinv, h_pair_1, h_pair_2, J_node, h_node

    @property
    def extra_expected_info_params(self):
        # TODO: Fix me!
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        _, _, _, logdet_pair = \
            self.dynamics_distn.meanfield_expectedstats()

        if self.diagonal_noise:
            # Use the fact that the diagonalregression prior is factorized
            _, _, E_sigmasq_inv, E_log_sigmasq = self.emission_distn.mf_expectations
            J_yy = E_sigmasq_inv
            logdet_node = -np.sum(E_log_sigmasq)

        else:
            raise NotImplementedError("Only supporting diagonal regression class right now")

        return J_init, h_init, logdet_pair, J_yy, logdet_node, self.data * self.mask

    def log_likelihood(self):
        if self._normalizer is None:
            self._normalizer, _, _ = kalman_info_filter(*self.info_params)

            # Update the normalization constant
            self._normalizer += self._extra_loglike_terms(
                self.A, self.B, self.sigma_states,
                self.C, self.D, self.sigma_obs,
                self.mu_init, self.sigma_init,
                self.inputs, self.mask * self.data,
                isdiag=self.diagonal_noise)

        return self._normalizer

    def filter(self):
        self._normalizer, self.filtered_mus, self.filtered_sigmas = \
            kalman_info_filter(*self.info_params)

        # Update the normalization constant
        self._normalizer += self._extra_loglike_terms(
            self.A, self.sigma_states, self.C, self.sigma_obs,
            self.mu_init, self.sigma_init, self.mask * self.data,
            isdiag=self.diagonal_noise)

    def smooth(self):
        if not hasattr(self, "smoothed_mus"):
            self.info_E_step()
        return self.smoothed_mus.dot(self.C.T) + self.inputs.dot(self.D.T)

    def info_E_step(self):
        self._normalizer, self.smoothed_mus, \
        self.smoothed_sigmas, E_xtp1_xtT = \
            info_E_step(*self.info_params)

        self._normalizer += self._extra_loglike_terms(
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.mu_init, self.sigma_init,
            self.inputs, self.mask * self.data,
            isdiag=self.diagonal_noise)

        self._set_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def resample(self):
        self._normalizer, self.stateseq = info_sample(*self.info_params)

        self._normalizer += self._extra_loglike_terms(
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.mu_init, self.sigma_init,
            self.inputs, self.mask * self.data,
            isdiag=self.diagonal_noise)

    def E_step(self):
        return self.info_E_step()

    def meanfieldupdate(self):
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(*self.expected_info_params)

        # TODO: Update the normalization code
        self._normalizer += self._info_extra_loglike_terms(
            *self.extra_expected_info_params,
            isdiag=self.diagonal_noise)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        # Get the emission stats
        p, n, d, T, mask, inputs, data = self.p, self.n, self.d, self.T, self.mask, self.inputs, self.data
        E_x_xT = smoothed_sigmas + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]
        E_x_uT = smoothed_mus[:,:,None] * self.inputs[:,None,:]
        E_u_uT = self.inputs[:,:,None] * self.inputs[:,None,:]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT,   E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0,2,1)), E_u_uT), axis=2)),
            axis=1)
        E_xut_xutT = E_xu_xuT[:-1].sum(0)

        E_xtp1_xtp1T = E_x_xT[1:].sum(0)
        E_xt_xtT = E_x_xT[:-1].sum(0)
        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        E_xtp1_utT = (smoothed_mus[1:,:,None] * inputs[:-1, None, :]).sum(0)
        E_xtp1_xutT = np.hstack((E_xtp1_xtT, E_xtp1_utT))


        def is_symmetric(A):
            return np.allclose(A, A.T)

        assert is_symmetric(E_xt_xtT)
        assert is_symmetric(E_xtp1_xtp1T)

        self.E_dynamics_stats = np.array(
            [E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, self.T - 1])

        # Emission statistics
        E_ysq = np.sum(data**2 * mask, axis=0)
        E_yxT = (data * mask).T.dot(smoothed_mus)
        E_yuT = (data * mask).T.dot(inputs)
        E_yxuT = np.hstack((E_yxT, E_yuT))
        E_xuxuT_vec = E_xu_xuT.reshape((T, -1))
        E_xuxuT = np.array([np.dot(self.mask[:, i], E_xuxuT_vec).reshape((n+d, n+d))
                          for i in range(p)])
        Tp = np.sum(self.mask, axis=0)

        self.E_emission_stats = objarray([E_ysq, E_yxuT, E_xuxuT, Tp])
