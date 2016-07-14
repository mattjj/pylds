from __future__ import division
import numpy as np

from pybasicbayes.util.general import AR_striding
from pybasicbayes.util.stats import mniw_expectedstats

from lds_messages_interface import kalman_filter, filter_and_sample, E_step, \
    info_E_step


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

        for t in xrange(1,T):
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
        for t in xrange(1,Tpred):
            states[t] = self.A.dot(states[t-1]) + \
                        self.B.dot(inputs[t-1]) + \
                        randseq[t-1]

        obs = states.dot(self.C.T) + inputs.dot(self.D.T)
        if obs_noise:
            L = np.linalg.cholesky(self.sigma_obs)
            obs += np.random.randn(Tpred, self.p).dot(L.T)

        return obs

    ### filtering

    def filter(self):
        self._normalizer, self.filtered_mus, self.filtered_sigmas = \
            kalman_filter(
                self.mu_init, self.sigma_init,
                self.A, self.B, self.sigma_states,
                self.C, self.D, self.sigma_obs,
                self.inputs, self.data)

    ### resampling

    def resample(self):
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

    def _set_expected_stats(self,smoothed_mus,smoothed_sigmas,E_xtp1_xtT):
        assert not np.isnan(E_xtp1_xtT).any()
        assert not np.isnan(smoothed_mus).any()
        assert not np.isnan(smoothed_sigmas).any()

        inputs, data = self.inputs, self.data

        # Now xx <- [x, u]
        EyyT = data.T.dot(data)
        EyxuT = data.T.dot(np.hstack((smoothed_mus, inputs)))
        # E[xx xx^T] =
        #  [[ E[xxT], E[xuT] ],
        #   [ E[uxT], E[uuT] ]]
        ExxT = smoothed_sigmas.sum(0) + smoothed_mus.T.dot(smoothed_mus)
        ExuT = smoothed_mus.T.dot(inputs)
        EuuT = inputs.T.dot(inputs)

        ExuxuT = np.asarray(
            np.bmat([[ExxT,   ExuT],
                     [ExuT.T, EuuT]]))

        # Account for the stats from all but the last time bin
        Exm1xm1T = \
            smoothed_sigmas[-1] + \
            np.outer(smoothed_mus[-1],smoothed_mus[-1])
        Exm1um1T = np.outer(smoothed_mus[-1], inputs[-1])
        Eum1um1T = np.outer(inputs[-1], inputs[-1])
        Exum1xum1T = \
            np.asarray(np.bmat(
                [[Exm1xm1T,   Exm1um1T],
                [Exm1um1T.T, Eum1um1T]]))

        E_xut_xutT = ExuxuT - Exum1xum1T

        # Account for the stats from all but the last time bin
        Ex0x0T = \
            smoothed_sigmas[0] + \
            np.outer(smoothed_mus[0], smoothed_mus[0])
        Ex0u0T = np.outer(smoothed_mus[0], inputs[0])
        Eu0u0T = np.outer(inputs[0], inputs[0])
        Exu0xu0T = \
            np.asarray(np.bmat(
                [[Ex0x0T, Ex0u0T],
                [Ex0u0T.T, Eu0u0T]]))

        E_xutp1_xutp1T = ExuxuT - Exu0xu0T

        # Compute the total statistics by summing over all time
        # E[(xp1, up1) (x, u)^T] =
        #  [[ E[xp1 xT], E[xp1 uT] ],
        #   [ E[up1 xT], E[up1 uT] ]]
        E_xtp1_xtT = E_xtp1_xtT.sum(0)
        E_xtp1_utT = smoothed_mus[1:].T.dot(inputs[:-1])
        E_utp1_xtT = inputs[1:].T.dot(smoothed_mus[:-1])
        E_utp1_utT = inputs[1:].T.dot(inputs[:-1])
        E_xutp1_xutT = \
            np.asarray(np.bmat(
                [[E_xtp1_xtT, E_xtp1_utT],
                 [E_utp1_xtT, E_utp1_utT]]))

        def is_symmetric(A):
            return np.allclose(A,A.T)

        assert is_symmetric(ExuxuT)
        assert is_symmetric(E_xut_xutT)
        assert is_symmetric(E_xutp1_xutp1T)

        self.E_emission_stats = np.array([EyyT, EyxuT, ExuxuT, self.T])
        self.E_dynamics_stats = np.array([E_xutp1_xutp1T[:self.n,:self.n], E_xutp1_xutT[:self.n,:], E_xut_xutT, self.T-1])

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

        J_node = C.T.dot(np.linalg.solve(sigma_obs, C))
        h_node = np.einsum('ik,ij,tj->tk',
                           C, np.linalg.inv(sigma_obs),
                           (data - inputs.dot(D.T)))

        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(
                J_init,h_init,
            J_pair_11,J_pair_21,J_pair_22,
            h_pair_1,h_pair_2,
            J_node,h_node)
        self._normalizer += self._extra_loglike_terms(
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.mu_init, self.sigma_init,
            self.inputs, self.data)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    @staticmethod
    def _extra_loglike_terms(A, B, sigma_states,
                             C, D, sigma_obs,
                             mu_init, sigma_init,
                             inputs, data):
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
        sigma_obs_inv = np.linalg.inv(sigma_obs)
        dt = inputs.dot(D.T)
        out -= 1./2 * np.einsum('ij,ti,tj->', sigma_obs_inv, data, data)
        out += np.einsum('ij,ti,tj->', sigma_obs_inv, data, dt)
        out -= 1./2 * np.einsum('ij,ti,tj->', sigma_obs_inv, dt, dt)
        out -= T/2. * np.linalg.slogdet(sigma_obs)[1]
        out -= T*p/2 * np.log(2*np.pi)

        return out

    ### mean field

    def meanfieldupdate(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        def get_params(distn):
            return mniw_expectedstats(
                *distn._natural_to_standard(distn.mf_natural_hypparam))

        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            get_params(self.dynamics_distn)

        # TODO: Check the logic behind these expectations.
        # Do we need to worry about correlations between A,B in E_BT_Qinv_A, for example?
        E_Qinv = J_pair_22
        E_AT_Qinv = J_pair_21[:,:self.n].T.copy("C")
        E_BT_Qinv = J_pair_21[:,self.n:].T
        E_BT_Qinv_A = E_BT_Qinv.dot(np.linalg.solve(E_Qinv, E_AT_Qinv.T))
        E_AT_Qinv_A = J_pair_11[:self.n, :self.n].copy("C")
        h_pair_1 = -self.inputs.dot(E_BT_Qinv_A)
        h_pair_2 = self.inputs.dot(E_BT_Qinv)

        J_yy, J_yx, J_node, logdet_node = get_params(self.emission_distn)
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
            J_init, h_init, logdet_pair, J_yy, logdet_node, data):
        # TODO: Update to compute terms with h_pair_1 and h_pair_2 if necessary
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

        # observation distribution
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
    def strided_stateseq(self):
        return AR_striding(self.stateseq,1)
