from __future__ import division
import numpy as np
from warnings import warn

from pybasicbayes.util.general import objarray
from pylds.lds_messages_interface import info_E_step, info_sample, kalman_info_filter, kalman_filter, E_step

import pypolyagamma as ppg
from pypolyagamma.distributions import _PGLogisticRegressionBase

# TODO on instantiating, maybe gaussian states should be resampled
# TODO make niter an __init__ arg instead of a method arg


###########
#  bases  #
###########

class _LDSStates(object):
    def __init__(self,model,T=None,data=None,inputs=None, stateseq=None,
            initialize_from_prior=False, initialize_to_noise=True):
        self.model = model

        self.T = T if T is not None else data.shape[0]
        self.data = data
        self.inputs = np.zeros((self.T, 0)) if inputs is None else inputs

        self._normalizer = None

        if stateseq is not None:
            self.gaussian_states = stateseq
        elif initialize_from_prior:
            self.generate_states()
        elif initialize_to_noise:
            self.gaussian_states = np.random.normal(size=(self.T, self.D_latent))
        elif data is not None:
            self.resample()
        else:
            raise Exception("Invalid options. Must specify how states are initialized.")

    ### Basics

    def log_likelihood(self):
        if self._normalizer is None:
            self._normalizer, _, _ = kalman_info_filter(*self.info_params)

            self._normalizer += self._info_extra_loglike_terms(
                *self.extra_info_params,
                isdiag=self.diagonal_noise)

        return self._normalizer

    def generate_states(self):
        # Generate from the prior and raise exception if unstable
        T, n = self.T, self.D_latent

        gss = np.empty((T,n),dtype='double')
        gss[0] = np.random.multivariate_normal(self.mu_init, self.sigma_init)

        for t in range(1,T):
            gss[t] = self.dynamics_distn.\
                rvs(x=np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])),
                    return_xy=False)
            assert np.all(np.isfinite(gss[t])), "LDS appears to be unstable!"

        self.gaussian_states = gss

    def generate_obs(self):
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T, p = self.T, self.D_emission
        ed = self.emission_distn
        gss = self.gaussian_states
        data = np.empty((T,p),dtype='double')

        for t in range(self.T):
            data[t] = \
                ed.rvs(x=np.hstack((gss[t][None, :], self.inputs[t][None,:])),
                       return_xy=False)

        return data

    def sample_predictions(self, Tpred, inputs=None, states_noise=False, obs_noise=False):
        inputs = np.zeros((Tpred, self.D_input)) if inputs is None else inputs
        # _, filtered_Js, filtered_hs = kalman_info_filter(*self.info_params)
        _, filtered_mus, filtered_sigmas = kalman_filter(
            self.mu_init, self.sigma_init,
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.inputs, self.data)

        init_mu = self.A.dot(filtered_mus[-1]) + self.B.dot(inputs[-1])
        init_sigma = self.sigma_states + self.A.dot(
            filtered_sigmas[-1]).dot(self.A.T)

        randseq = np.zeros((Tpred - 1, self.D_latent))
        if states_noise:
            L = np.linalg.cholesky(self.sigma_states)
            randseq += np.random.randn(Tpred - 1, self.D_latent).dot(L.T)

        states = np.empty((Tpred, self.D_latent))
        states[0] = np.random.multivariate_normal(init_mu, init_sigma)
        for t in range(1, Tpred):
            states[t] = self.A.dot(states[t - 1]) + \
                        self.B.dot(inputs[t - 1]) + \
                        randseq[t - 1]

        obs = states.dot(self.C.T) + inputs.dot(self.D.T)
        if obs_noise:
            L = np.linalg.cholesky(self.sigma_obs)
            obs += np.random.randn(Tpred, self.D_emission).dot(L.T)

        return obs

    ## convenience properties

    @property
    def D_latent(self):
        return self.dynamics_distn.D_out

    @property
    def D_input(self):
        return self.dynamics_distn.D_in - self.dynamics_distn.D_out

    @property
    def D_emission(self):
        return self.emission_distn.D_out

    @property
    def dynamics_distn(self):
        return self.model.dynamics_distn

    @property
    def emission_distn(self):
        return self.model.emission_distn

    @property
    def diagonal_noise(self):
        return self.model.diagonal_noise

    @property
    def mu_init(self):
        return self.model.mu_init

    @property
    def sigma_init(self):
        return self.model.sigma_init

    @property
    def A(self):
        return self.dynamics_distn.A[:, :self.D_latent]

    @property
    def B(self):
        return self.dynamics_distn.A[:, self.D_latent:]

    @property
    def sigma_states(self):
        return self.dynamics_distn.sigma

    @property
    def C(self):
        return self.emission_distn.A[:,:self.D_latent]

    @property
    def D(self):
        return self.emission_distn.A[:, self.D_latent:]

    @property
    def sigma_obs(self):
        return self.emission_distn.sigma

    @property
    def _kwargs(self):
        return dict(super(_LDSStates, self)._kwargs,
                    gaussian_states=self.gaussian_states)

    @property
    def info_init_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)
        return J_init, h_init

    @property
    def info_dynamics_params(self):
        A = self.A
        B = self.B
        Q = self.sigma_states

        # Get the pairwise potentials
        # TODO: Check for diagonal before inverting
        J_pair_22 = np.linalg.inv(Q)
        J_pair_21 = -J_pair_22.dot(A)
        J_pair_11 = A.T.dot(-J_pair_21)

        # Check if diagonal and avoid inverting D_obs x D_obs matrix
        mBTQiA = B.T.dot(J_pair_21)
        BTQi = B.T.dot(J_pair_22)
        h_pair_1 = self.inputs.dot(mBTQiA)
        h_pair_2 = self.inputs.dot(BTQi)

        return J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2

    @property
    def info_emission_params(self):
        # TODO: Check for diagonal emissions
        C = self.C
        D = self.D
        R = self.sigma_obs
        RinvC = np.linalg.solve(R, C)
        J_node = C.T.dot(RinvC)

        # TODO: Faster to replace this with a loop?
        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        h_node = (self.data - self.inputs.dot(D.T)).dot(RinvC)

        return J_node, h_node

    @property
    def info_params(self):
        return self.info_init_params + self.info_dynamics_params + self.info_emission_params

    @property
    def extra_info_params(self):
        # TODO: This should have terms related to self.inputs
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        Q = self.sigma_states
        logdet_pair = -np.linalg.slogdet(Q)[1]

        # We need terms for u_t B^T Q^{-1} B u
        B = self.B
        hJh_pair = B.T.dot(np.linalg.solve(Q, B))

        # Observations
        if self.diagonal_noise:
            # Use the fact that the diagonalregression prior is factorized
            rsq = self.emission_distn.sigmasq_flat
            J_yy = 1./rsq
            logdet_node = -np.sum(np.log(rsq))

            # We need terms for u_t D^T R^{-1} D u
            D = self.D
            hJh_node = D.T.dot(np.diag(J_yy).dot(D))

        else:
            R = self.sigma_obs
            J_yy = np.linalg.inv(R)
            logdet_node = -np.linalg.slogdet(R)[1]

            # We need terms for u_t D^T R^{-1} D u
            D = self.D
            hJh_node = D.T.dot(np.linalg.solve(R, D))

        return J_init, h_init, logdet_pair, hJh_pair, J_yy, logdet_node, hJh_node, self.inputs, self.data

    @staticmethod
    def _info_extra_loglike_terms(
            J_init, h_init, logdet_pair, hJh_pair, J_yy, logdet_node, hJh_node, inputs, data,
            isdiag=False):
        warn("Log likelihood calculations are not correct!")
        p, n, T = J_yy.shape[0], h_init.shape[0], data.shape[0]

        out = 0.

        # Initial distribution
        out -= 1./2 * h_init.dot(np.linalg.solve(J_init, h_init))
        out += 1./2 * np.linalg.slogdet(J_init)[1]
        out -= n/2. * np.log(2*np.pi)

        # -1/2 log |Q| -n/2 log(2pi)
        out += 1./2 * logdet_pair.sum() if isinstance(logdet_pair, np.ndarray) \
            else (T-1)/2. * logdet_pair
        out -= (T-1)*n/2. * np.log(2*np.pi)

        # We need terms for u_t B^T Q^{-1} B u
        contract = 'ij,ti,tj->' if hJh_pair.ndim == 2 else 'tij,ti,tj->'
        out -= 1. / 2 * np.einsum(contract, hJh_pair, inputs[:-1], inputs[:-1])

        # -1/2 y^T R^{-1} y
        if isdiag:
            assert (J_yy.ndim == 1 and J_yy.shape[0] == data.shape[1]) or \
                   (J_yy.shape == data.shape)
            out -= 1./2 * np.sum(data**2 * J_yy)
        else:
            contract = 'ij,ti,tj->' if J_yy.ndim == 2 else 'tij,ti,tj->'
            out -= 1./2 * np.einsum(contract, J_yy, data, data)

        # We need terms for u_t D^T R^{-1} D u
        contract = 'ij,ti,tj->' if hJh_node.ndim == 2 else 'tij,ti,tj->'
        out -= 1. / 2 * np.einsum(contract, hJh_node, inputs, inputs)

        # -1/2 log |R| -p/2 log(2 pi)
        out += 1./2 * logdet_node.sum() if isinstance(logdet_node, np.ndarray) \
            else T/2. * logdet_node
        out -= T*p/2. * np.log(2*np.pi)

        return out


    def filter(self):
        # self._normalizer, self.filtered_Js, self.filtered_hs = \
        #     kalman_info_filter(*self.info_params)

        _, filtered_mus, filtered_sigmas = kalman_filter(
            self.mu_init, self.sigma_init,
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.inputs, self.data)

        # Update the normalization constant
        self._gaussian_normalizer += self._info_extra_loglike_terms(
            *self.extra_info_params,
            isdiag=self.diagonal_noise)

    def smooth(self):
        # Use the info E step because it can take advantage of diagonal noise
        # The standard E step could but we have not implemented it
        self.info_E_step()
        return self.smoothed_mus.dot(self.C.T) + self.inputs.dot(self.D.T)

    ### Expectations
    def E_step(self):
        return self.info_E_step()

    def std_E_step(self):
        # TODO: Update normalizer?
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
        E_xtp1_xtT = E_step(
            self.mu_init, self.sigma_init,
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.inputs, self.data)

        self._set_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def info_E_step(self):
        self._normalizer, self.smoothed_mus, \
        self.smoothed_sigmas, E_xtp1_xtT = \
            info_E_step(*self.info_params)

        self._normalizer += self._info_extra_loglike_terms(
            *self.extra_info_params,
            isdiag=self.diagonal_noise)

        self._set_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        # Get the emission stats
        p, n, d, T, inputs, data = \
            self.D_emission, self.D_latent, self.D_input, self.T, \
            self.inputs, self.data

        E_x_xT = smoothed_sigmas + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]
        E_x_uT = smoothed_mus[:, :, None] * self.inputs[:, None, :]
        E_u_uT = self.inputs[:, :, None] * self.inputs[:, None, :]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT, E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0, 2, 1)), E_u_uT), axis=2)),
            axis=1)
        E_xut_xutT = E_xu_xuT[:-1].sum(0)

        E_xtp1_xtp1T = E_x_xT[1:].sum(0)
        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        E_xtp1_utT = (smoothed_mus[1:, :, None] * inputs[:-1, None, :]).sum(0)
        E_xtp1_xutT = np.hstack((E_xtp1_xtT, E_xtp1_utT))

        # def is_symmetric(A):
        #     return np.allclose(A, A.T)
        # assert is_symmetric(E_xt_xtT)
        # assert is_symmetric(E_xtp1_xtp1T)

        self.E_dynamics_stats = np.array(
            [E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, self.T - 1])

        # Emission statistics
        E_yyT = np.sum(data**2, axis=0) if self.diagonal_noise else data.T.dot(data)
        E_yxT = data.T.dot(smoothed_mus)
        E_yuT = data.T.dot(inputs)
        E_yxuT = np.hstack((E_yxT, E_yuT))

        self.E_emission_stats = objarray([E_yyT, E_yxuT, E_xu_xuT.sum(0), T])

######################
#  algorithm mixins  #
######################

class _LDSStatesGibbs(_LDSStates):
    def resample(self, niter=1):
        self.resample_gaussian_states()

    def _init_gibbs_from_mf(self):
        raise NotImplementedError  # TODO

    def resample_gaussian_states(self):
        self._normalizer, self.gaussian_states = \
            info_sample(*self.info_params)
        self._normalizer += self._info_extra_loglike_terms(
            *self.extra_info_params, isdiag=self.diagonal_noise)

class _LDSStatesMeanField(_LDSStates):
    @property
    def expected_info_dynamics_params(self):
        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            self.dynamics_distn.meanfield_expectedstats()

        # Compute E[B^T Q^{-1}] and E[B^T Q^{-1} A]
        n = self.D_latent
        E_Qinv = J_pair_22.copy("C")
        E_AT_Qinv = (J_pair_21[:,:n].T).copy("C")
        E_BT_Qinv = (J_pair_21[:,n:].T).copy("C")
        E_BT_Qinv_A = J_pair_11[n:,:n].copy("C")
        E_AT_Qinv_A = J_pair_11[:n,:n].copy("C")

        h_pair_1 = (-self.inputs.dot(E_BT_Qinv_A)).copy("C")
        h_pair_2 = (self.inputs.dot(E_BT_Qinv)).copy("C")

        return E_AT_Qinv_A, -E_AT_Qinv, E_Qinv, h_pair_1, h_pair_2

    @property
    def expected_info_emission_params(self):
        # TODO: Use the fact that the diagonalregression prior is factorized
        # E_C, E_CCT, E_sigmasq_inv, _ = self.emission_distn.mf_expectations
        # E_C, E_D = E_C[:, :self.n], E_C[:, self.n:]
        # E_CCT_vec = E_CCT.reshape(self.p, -1)
        #
        # J_obs = self.mask * E_sigmasq_inv
        # J_node = (np.dot(J_obs, E_CCT_vec)).reshape((self.T, self.n, self.n))
        # h_node = (self.data * J_obs).dot(E_C) - (self.inputs.dot(E_D.T) * J_obs).dot(E_C)

        J_yy, J_yx, J_node, logdet_node = \
            self.emission_distn.meanfield_expectedstats()

        n = self.D_latent
        E_Rinv_C = J_yx[:,:n].copy("C")
        E_DT_Rinv_C = (J_node[n:,:n]).copy("C")
        E_CT_Rinv_C = J_node[:n,:n].copy("C")

        h_node = self.data.dot(E_Rinv_C)
        h_node -= self.inputs.dot(E_DT_Rinv_C)

        return E_CT_Rinv_C, h_node

    @property
    def expected_info_params(self):
        return self.info_init_params + \
               self.expected_info_dynamics_params + \
               self.expected_info_emission_params

    @property
    def expected_extra_info_params(self):
        n = self.D_latent
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        _, _, J_pair_11, logdet_pair = self.dynamics_distn.meanfield_expectedstats()
        hJh_pair = J_pair_11[n:, n:]

        # Observations
        if self.diagonal_noise:
            _, mf_E_CCT, mf_E_sigmasq_inv, mf_E_log_sigmasq = self.emission_distn.mf_expectations
            J_yy = mf_E_sigmasq_inv
            logdet_node = -np.sum(mf_E_log_sigmasq)

            mf_E_DDT = mf_E_CCT[:,n:,n:]
            hJh_node = np.sum(mf_E_DDT * mf_E_sigmasq_inv[:,None,None], axis=0)
        else:
            J_yy, _, J_node, logdet_node = \
                self.emission_distn.meanfield_expectedstats()
            hJh_node = J_node[n:, n:]

        return J_init, h_init, logdet_pair, hJh_pair, J_yy, logdet_node, hJh_node, self.inputs, self.data

    def meanfieldupdate(self):
        self._mf_lds_normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(*self.expected_info_params)

        self._mf_lds_normalizer += LDSStates._info_extra_loglike_terms(
            *self.expected_extra_info_params, isdiag=self.diagonal_noise)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def get_vlb(self):
        return self._mf_lds_normalizer

    def meanfield_smooth(self):
        if self.diagonal_noise:
            E_C, _, _, _ = self.emission_distn.mf_expectations
        else:
            ed = self.emission_distn
            _,_,E_C,_ = ed._natural_to_standard(ed.mf_natural_hypparam)
        return np.hstack((self.smoothed_mus, self.inputs)).dot(E_C.T)


class _LDSStatesMaskedData(_LDSStatesGibbs, _LDSStatesMeanField):
    def __init__(self, model, data=None, mask=None, **kwargs):
        if mask is not None:
            assert mask.shape == data.shape
            self.mask = mask
        elif data is not None and np.any(np.isnan(data)):
            warn("data includes NaN's. Treating these as missing data.")
            self.mask = ~np.isnan(data)
            # TODO: We should make this unnecessary
            warn("zeroing out nans in data to make sure code works")
            data[np.isnan(data)] = 0
        else:
            self.mask = None

        super(_LDSStatesMaskedData, self).__init__(model, data=data, **kwargs)

    @property
    def info_emission_params(self):
        if self.mask is None:
            return super(_LDSStatesMaskedData, self).info_emission_params

        if self.diagonal_noise:
            return self._info_emission_params_diag
        else:
            return self._info_emission_params_dense

    @property
    def _info_emission_params_diag(self):
        C, D = self.C, self.D
        sigmasq = self.emission_distn.sigmasq_flat
        J_obs = self.mask / sigmasq

        CCT = np.array([np.outer(cp, cp) for cp in C]).\
            reshape((self.D_emission, self.D_latent ** 2))

        J_node = np.dot(J_obs, CCT)
        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        h_node = (self.data * J_obs).dot(C)
        if self.D_input > 0:
            h_node -= (self.inputs.dot(D.T) * J_obs).dot(C)

        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))
        return J_node, h_node

    @property
    def _info_emission_params_dense(self):
        T, D_latent = self.T, self.D_latent
        data, inputs, mask = self.data, self.inputs, self.mask

        C, D, R = self.C, self.D, self.sigma_obs
        Rinv = np.linang.inv(R)

        # Sloowwwwww
        J_node = np.zeros((T, D_latent, D_latent))
        h_node = np.zeros((T, D_latent, D_latent))
        for t in range(T):
            Rinv_t = Rinv * np.outer(mask[t], mask[t])
            J_node[t] = C.T.dot(Rinv_t).dot(C)
            h_node[t] = (data[t] - inputs[t].dot(D.T)).dot(Rinv_t).dot(C)

        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))
        return J_node, h_node

    @property
    def extra_info_params(self):
        params = super(_LDSStatesMaskedData, self).extra_info_params

        # TODO: Mask off missing data entries -- might not work?
        if self.mask is not None:
            params = params[:-1] + (self.data * self.mask, )

        return params

    @property
    def expected_info_emission_params(self):
        n = self.D_latent
        if self.mask is None:
            return super(_LDSStatesMaskedData, self).expected_info_emission_params

        if self.diagonal_noise:
            E_C, E_CCT, E_sigmasq_inv, _ = self.emission_distn.mf_expectations
            E_C, E_D = E_C[:,:n], E_C[:,n:]
            J_obs = self.mask * E_sigmasq_inv

            J_node = np.dot(J_obs, E_CCT)
            h_node = (self.data * J_obs).dot(E_C)
            h_node -= (self.inputs.dot(E_D.T) * J_obs).dot(E_C)

            J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))

        else:
            raise NotImplementedError("Only supporting diagonal regression class with missing data now")

        return J_node, h_node

    @property
    def expected_extra_info_params(self):
        params = super(_LDSStatesMaskedData, self).expected_extra_info_params

        # Mask off missing data entries -- should work?
        if self.mask is not None:
            params = params[:-1] + (self.data * self.mask, )

        return params

    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        if self.mask is None:
            return super(_LDSStatesMaskedData, self).\
                _set_expected_stats(smoothed_mus, smoothed_sigmas, E_xtp1_xtT)

        # Get the emission stats
        p, n, d, T, mask, inputs, data = \
            self.D_emission, self.D_latent, self.D_input, self.T, \
            self.mask, self.inputs, self.data
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


class _LDSStatesCountData(_LDSStatesMaskedData, _LDSStatesGibbs):
    def __init__(self, model, data=None, mask=None, **kwargs):
        super(_LDSStatesCountData, self). \
            __init__(model, data=data, mask=mask, **kwargs)

        # Check if the emission matrix is a count regression
        if isinstance(self.emission_distn, _PGLogisticRegressionBase):
            self.has_count_data = True

            # Initialize the Polya-gamma samplers
            num_threads = ppg.get_omp_num_threads()
            seeds = np.random.randint(2 ** 16, size=num_threads)
            self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

            # Initialize auxiliary variables, omega
            self.omega = np.ones((self.T, self.D_emission), dtype=np.float)
        else:
            self.has_count_data = False

    @property
    def sigma_obs(self):
        if self.has_count_data:
            raise Exception("Count data does not have sigma_obs")
        return super(_LDSStatesCountData, self).sigma_obs

    @property
    def info_emission_params(self):
        if not self.has_count_data:
            return super(_LDSStatesCountData, self).info_emission_params


        # Otherwise, use the Polya-gamma augmentation
        # log p(y_{tn} | x, om)
        #   = -0.5 * om_{tn} * (c_n^T x_t + d_n^T u_t + b_n)**2
        #     + kappa * (c_n * x_t + d_n^Tu_t + b_n)
        #   = -0.5 * om_{tn} * (x_t^T c_n c_n^T x_t
        #                       + 2 x_t^T c_n d_n^T u_t
        #                       + 2 x_t^T c_n b_n)
        #     + x_t^T (kappa_{tn} * c_n)
        #   = -0.5 x_t^T (c_n c_n^T * om_{tn}) x_t
        #     +  x_t^T * (kappa_{tn} - d_n^T u_t * om_{tn} -b_n * om_{tn}) * c_n
        #
        # Thus
        # J = (om * mask).dot(CCT)
        # h = ((kappa - om * d) * mask).dot(C)
        T, D_latent, D_emission = self.T, self.D_latent, self.D_emission
        data, inputs, mask, omega = self.data, self.inputs, self.mask, self.omega
        # TODO: This is hacky...
        mask = self.mask if self.mask is not None else np.ones_like(self.data)
        emission_distn = self.emission_distn

        C = emission_distn.A[:, :D_latent]
        D = emission_distn.A[:,D_latent:]
        b = emission_distn.b
        CCT = np.array([np.outer(cp, cp) for cp in C]).\
            reshape((D_emission, D_latent ** 2))

        J_node = np.dot(omega * mask, CCT)
        J_node = J_node.reshape((T, D_latent, D_latent))

        kappa = emission_distn.kappa_func(data)
        h_node = ((kappa - omega * b.T - omega * inputs.dot(D.T)) * mask).dot(C)
        return J_node, h_node

    @property
    def extra_info_params(self):
        if not self.has_count_data:
            return super(_LDSStatesCountData, self).extra_info_params

        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        Q = self.sigma_states
        logdet_pair = -np.linalg.slogdet(Q)[1]

        # TODO: Observations
        warn("extra_info_params not implemented for count data. "
             "Log likelihood calculations will be wrong.")
        J_yy = np.zeros((self.D_emission, self.D_emission))
        logdet_node = np.zeros((self.T))

        masked_data = self.data * self.mask \
            if self.mask is not None \
            else self.data

        return J_init, h_init, logdet_pair, J_yy, logdet_node, masked_data

    @property
    def expected_info_emission_params(self):
        if self.has_count_data:
            raise NotImplementedError("Mean field with count observations is not yet supported")

        return super(_LDSStatesCountData, self).expected_info_emission_params

    @property
    def expected_extra_info_params(self):
        if self.has_count_data:
            raise NotImplementedError("Mean field with count observations is not yet supported")

        return super(_LDSStatesCountData, self).expected_extra_info_params

    def resample(self, niter=1):
        self.resample_gaussian_states()

        if self.has_count_data:
            self.resample_auxiliary_variables()

    def resample_auxiliary_variables(self):
        C, D, ed = self.C, self.D, self.emission_distn
        psi = self.gaussian_states.dot(C.T) + self.inputs.dot(D.T) + ed.b.T

        b = ed.b_func(self.data)
        ppg.pgdrawvpar(self.ppgs, b.ravel(), psi.ravel(), self.omega.ravel())

    def smooth(self):
        if not self.has_count_data:
            return super(_LDSStatesCountData, self).smooth()

        X = np.column_stack((self.gaussian_states, self.inputs))
        mean = self.emission_distn.mean(X)

        return mean


####################
#  states classes  #
####################

class LDSStates(
    _LDSStatesCountData,
    _LDSStatesMaskedData,
    _LDSStatesGibbs,
    _LDSStatesMeanField):
    pass

