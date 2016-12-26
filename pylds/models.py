from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix

from pybasicbayes.abstractions import Model, ModelGibbsSampling, \
    ModelEM, ModelMeanField, ModelMeanFieldSVI

from pybasicbayes.distributions import DiagonalRegression, Gaussian, Regression
from pylds.distributions import PoissonRegression
from pylds.states import LDSStates, LDSStatesZeroInflatedCountData, LaplaceApproxPoissonLDSStates
from pylds.util import random_rotation

# NOTE: dynamics_distn should be an instance of Regression,
# emission_distn should be an instance of Regression, and
# init_dynamics_distn should probably be an instance of Gaussian

class _LDSBase(Model):

    _states_class = LDSStates

    def __init__(self,dynamics_distn,emission_distn):
        self.dynamics_distn = dynamics_distn
        self.emission_distn = emission_distn
        self.states_list = []

    def add_data(self,data, inputs=None, mask=None, **kwargs):
        assert isinstance(data,np.ndarray)
        self.states_list.append(self._states_class(model=self, data=data, inputs=inputs, mask=mask, **kwargs))
        return self

    def log_likelihood(self, data=None, inputs=None, mask=None):
        if data is not None:
            assert isinstance(data,(list,np.ndarray))
            if isinstance(data,np.ndarray):
                self.add_data(data=data, inputs=inputs, mask=mask)
                return self.states_list.pop().log_likelihood()
            else:
                return sum(self.log_likelihood(d, i) for (d, i) in zip(data, inputs))
        else:
            return sum(s.log_likelihood() for s in self.states_list)

    def generate(self, T, inputs=None, keep=True):
        s = self._states_class(model=self, T=T, inputs=inputs, initialize_from_prior=True)
        data = self._generate_obs(s, inputs)
        if keep:
            self.states_list.append(s)
        return data, s.gaussian_states

    def _generate_obs(self,s, inputs):
        if s.data is None:
            inputs = np.zeros((s.T, 0)) if inputs is None else inputs
            s.data = self.emission_distn.rvs(
                x=np.hstack((s.gaussian_states, inputs)), return_xy=False)
        else:
            # filling in missing data
            raise NotImplementedError
        return s.data

    def smooth(self, data, inputs=None, mask=None):
        self.add_data(data, inputs=inputs, mask=mask)
        s = self.states_list.pop()
        return s.smooth()

    def predict(self, data, Tpred):
        # return means and covariances
        raise NotImplementedError

    def sample_predictions(self, data, Tpred, inputs_pred=None, inputs=None, mask=None, states_noise=True, obs_noise=True):
        self.add_data(data, inputs=inputs, mask=mask)
        s = self.states_list.pop()
        return s.sample_predictions(Tpred, inputs=inputs_pred, states_noise=states_noise, obs_noise=obs_noise)

    # convenience properties

    @property
    def D_latent(self):
        'latent dimension'
        return self.dynamics_distn.D_out

    @property
    def D_obs(self):
        'emission dimension'
        return self.emission_distn.D_out

    @property
    def D_input(self):
        'input dimension'
        return self.dynamics_distn.D_in - self.dynamics_distn.D_out

    @property
    def mu_init(self):
        return np.zeros(self.D_latent) if not hasattr(self, '_mu_init') \
            else self._mu_init

    @mu_init.setter
    def mu_init(self,mu_init):
        self._mu_init = mu_init

    @property
    def sigma_init(self):
        if hasattr(self,'_sigma_init'):
            return self._sigma_init

        try:
            from scipy.linalg import solve_discrete_lyapunov as dtlyap
            return dtlyap(self.A, self.sigma_states)
        except ImportError:
            return np.linalg.solve(
                np.eye(self.D_latent ** 2) - np.kron(self.A, self.A), self.sigma_states.ravel())\
                .reshape(self.D_latent, self.D_latent)

    @sigma_init.setter
    def sigma_init(self,sigma_init):
        self._sigma_init = sigma_init

    @property
    def A(self):
        return self.dynamics_distn.A[:, :self.D_latent].copy("C")

    @A.setter
    def A(self,A):
        self.dynamics_distn.A[:, :self.D_latent] = A

    @property
    def B(self):
        return self.dynamics_distn.A[:, self.D_latent:].copy("C")

    @B.setter
    def B(self, B):
        self.dynamics_distn.A[:, self.D_latent:] = B

    @property
    def sigma_states(self):
        return self.dynamics_distn.sigma

    @sigma_states.setter
    def sigma_states(self,sigma_states):
        self.dynamics_distn.sigma = sigma_states

    @property
    def C(self):
        return self.emission_distn.A[:, :self.D_latent].copy("C")

    @C.setter
    def C(self,C):
        self.emission_distn.A[:, :self.D_latent] = C

    @property
    def D(self):
        return self.emission_distn.A[:, self.D_latent:].copy("C")

    @D.setter
    def D(self, D):
        self.emission_distn.A[:, self.D_latent:] = D

    @property
    def sigma_obs(self):
        return self.emission_distn.sigma

    @sigma_obs.setter
    def sigma_obs(self,sigma_obs):
        self.emission_distn.sigma = sigma_obs

    @property
    def diagonal_noise(self):
        return isinstance(self.emission_distn, DiagonalRegression)

    @property
    def sigma_obs_flat(self):
        return self.emission_distn.sigmasq_flat

    @sigma_obs_flat.setter
    def sigma_obs_flat(self, value):
        self.emission_distn.sigmasq_flat = value

    @property
    def is_stable(self):
        return np.max(np.abs(np.linalg.eigvals(self.dynamics_distn.A))) < 1.

    @property
    def has_missing_data(self):
        return any([s.mask is not None for s in self.states_list])

    @property
    def has_count_data(self):
        return any([hasattr(s, "omega") and s.omega is not None for s in self.states_list])


class _LDSGibbsSampling(_LDSBase, ModelGibbsSampling):
    def resample_model(self):
        self.resample_parameters()
        self.resample_states()

    def resample_states(self):
        for s in self.states_list:
            s.resample()

    def resample_parameters(self):
        self.resample_dynamics_distn()
        self.resample_emission_distn()

    def resample_dynamics_distn(self):
        self.dynamics_distn.resample(
            [np.hstack((s.gaussian_states[:-1],s.inputs[:-1],s.gaussian_states[1:]))
             for s in self.states_list])

    def resample_emission_distn(self):
        xys = [(np.hstack((s.gaussian_states, s.inputs)), s.data) for s in self.states_list]
        mask = [s.mask for s in self.states_list] if self.has_missing_data else None
        omega = [s.omega for s in self.states_list] if self.has_count_data else None

        if self.has_count_data:
            self.emission_distn.resample(data=xys, mask=mask, omega=omega)
        elif self.has_missing_data:
            self.emission_distn.resample(data=xys, mask=mask)
        else:
            self.emission_distn.resample(data=xys)

class _LDSMeanField(_LDSBase, ModelMeanField):
    def meanfield_coordinate_descent_step(self):
        for s in self.states_list:
            if not hasattr(s, 'E_emission_stats'):
                s.meanfieldupdate()

        self.meanfield_update_parameters()
        self.meanfield_update_states()

        return self.vlb()

    def meanfield_update_states(self):
        for s in self.states_list:
            s.meanfieldupdate()

    def meanfield_update_parameters(self):
        self.meanfield_update_dynamics_distn()
        self.meanfield_update_emission_distn()

    def meanfield_update_dynamics_distn(self):
        self.dynamics_distn.meanfieldupdate(
            stats=(sum(s.E_dynamics_stats for s in self.states_list)))

    def meanfield_update_emission_distn(self):
        self.emission_distn.meanfieldupdate(
            stats=(sum(s.E_emission_stats for s in self.states_list)))

    def resample_from_mf(self):
        self.dynamics_distn.resample_from_mf()
        self.emission_distn.resample_from_mf()

    def vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += self.emission_distn.get_vlb()
        vlb += self.dynamics_distn.get_vlb()
        return vlb


class _LDSMeanFieldSVI(_LDSBase, ModelMeanFieldSVI):
    def meanfield_sgdstep(self, minibatch, prob, stepsize, masks=None, **kwargs):
        states_list = self._get_mb_states_list(minibatch, masks, **kwargs)
        for s in states_list:
            s.meanfieldupdate()
        self._meanfield_sgdstep_parameters(states_list, prob, stepsize)

    def _meanfield_sgdstep_parameters(self, states_list, prob, stepsize):
        self._meanfield_sgdstep_dynamics_distn(states_list, prob, stepsize)
        self._meanfield_sgdstep_emission_distn(states_list, prob, stepsize)

    def _meanfield_sgdstep_dynamics_distn(self, states_list, prob, stepsize):
        self.dynamics_distn.meanfield_sgdstep(
            data=None, weights=None,
            stats=(sum(s.E_dynamics_stats for s in states_list)),
            prob=prob, stepsize=stepsize)

    def _meanfield_sgdstep_emission_distn(self, states_list, prob, stepsize):
        self.emission_distn.meanfield_sgdstep(
            data=None, weights=None,
            stats=(sum(s.E_emission_stats for s in states_list)),
            prob=prob, stepsize=stepsize)

    def _get_mb_states_list(self, minibatch, masks, **kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
        masks = [None] * len(minibatch) if masks is None else \
            (masks if isinstance(masks, list) else [masks])

        def get_states(data, mask):
            self.add_data(data, mask=mask, **kwargs)
            return self.states_list.pop()

        return [get_states(data, mask) for data, mask in zip(minibatch, masks)]


class _NonstationaryLDSGibbsSampling(_LDSGibbsSampling):
    def resample_model(self):
        self.resample_init_dynamics_distn()
        super(_NonstationaryLDSGibbsSampling, self).resample_model()

    def resample_init_dynamics_distn(self):
        self.init_dynamics_distn.resample(
            [s.gaussian_states[0] for s in self.states_list])


class _LDSEM(_LDSBase, ModelEM):
    def EM_step(self):
        self.E_step()
        self.M_step()

    def E_step(self):
        for s in self.states_list:
            s.E_step()

    def M_step(self):
        self.M_step_dynamics_distn()
        self.M_step_emission_distn()

    def M_step_dynamics_distn(self):
        self.dynamics_distn.max_likelihood(
            data=None,
            stats=(sum(s.E_dynamics_stats for s in self.states_list)))

    def M_step_emission_distn(self):
        self.emission_distn.max_likelihood(
            data=None,
            stats=(sum(s.E_emission_stats for s in self.states_list)))


class _NonstationaryLDSEM(_LDSEM):
    def M_Step(self):
        self.M_step_init_dynamics_distn()
        super(_NonstationaryLDSEM, self).M_step()

    def M_step_init_dynamics_distn(self):
        self.init_dynamics_distn.max_likelihood(
            stats=(sum(s.E_x1_x1 for s in self.states_list)))


###################
#  model classes  #
###################

class LDS(_LDSGibbsSampling, _LDSMeanField, _LDSEM, _LDSMeanFieldSVI, _LDSBase):
    pass


class NonstationaryLDS(
        _NonstationaryLDSGibbsSampling,
        _NonstationaryLDSEM,
        _LDSBase):
    def __init__(self, init_dynamics_distn, *args, **kwargs):
        self.init_dynamics_distn = init_dynamics_distn
        super(NonstationaryLDS, self).__init__(*args, **kwargs)

    def resample_init_dynamics_distn(self):
        self.init_dynamics_distn.resample(
            [s.gaussian_states[0] for s in self.states_list])

    # convenience properties

    @property
    def mu_init(self):
        return self.init_dynamics_distn.mu

    @mu_init.setter
    def mu_init(self, mu_init):
        self.init_dynamics_distn.mu = mu_init

    @property
    def sigma_init(self):
        return self.init_dynamics_distn.sigma

    @sigma_init.setter
    def sigma_init(self, sigma_init):
        self.init_dynamics_distn.sigma = sigma_init

class ZeroInflatedCountLDS(_LDSGibbsSampling, _LDSBase):
    _states_class = LDSStatesZeroInflatedCountData

    def __init__(self, rho, *args, **kwargs):
        """
        :param rho: Probability of count drawn from model
                    With pr 1-rho, the emission is deterministically zero
        """
        super(ZeroInflatedCountLDS, self).__init__(*args, **kwargs)
        self.rho = rho

    def add_data(self,data, inputs=None, mask=None, **kwargs):
        self.states_list.append(self._states_class(model=self, data=data, inputs=inputs, mask=mask, **kwargs))
        return self

    def _generate_obs(self,s, inputs):
        if s.data is None:
            # TODO: Do this sparsely
            inputs = np.zeros((s.T, 0)) if inputs is None else inputs
            data = self.emission_distn.rvs(
                x=np.hstack((s.gaussian_states, inputs)), return_xy=False)

            # Zero out data
            zeros = np.random.rand(s.T, self.D_obs) > self.rho
            data[zeros] = 0
            s.data = csr_matrix(data)

        else:
            # filling in missing data
            raise NotImplementedError
        return s.data

    def resample_emission_distn(self):
        """
        Now for the expensive part... the data is stored in a sparse row
        format, which is good for updating the latent states (since we
        primarily rely on dot products with the data, which can be
        efficiently performed for CSR matrices).

        However, in order to update the n-th row of the emission matrix,
        we need to know which counts are observed in the n-th column of data.
        This involves converting the data to a sparse column format, which
        can require (time) intensive re-indexing.
        """
        masked_datas = [s.masked_data.tocsc() for s in self.states_list]
        xs = [np.hstack((s.gaussian_states, s.inputs))for s in self.states_list]

        for n in range(self.D_obs):
            # Get the nonzero values of the nth column
            rowns = [md.indices[md.indptr[n]:md.indptr[n+1]] for md in masked_datas]
            xns = [x[r] for x,r in zip(xs, rowns)]
            yns = [s.masked_data.getcol(n).data for s in self.states_list]
            maskns = [np.ones_like(y, dtype=bool) for y in yns]
            omegans = [s.omega.getcol(n).data for s in self.states_list]
            self.emission_distn._resample_row_of_emission_matrix(n, xns, yns, maskns, omegans)


class LaplaceApproxPoissonLDS(NonstationaryLDS, _NonstationaryLDSEM):
    _states_class = LaplaceApproxPoissonLDSStates
    @property
    def d(self):
        return self.emission_distn.b

    @d.setter
    def d(self, value):
        self.emission_distn.b = value

    def mode_log_likelihood(self):
        return sum(s.mode_log_likelihood() for s in self.states_list)

    def heldout_log_likelihood(self):
        return sum(s.heldout_log_likelihood() for s in self.states_list)

    def mode_heldout_log_likelihood(self):
        return sum(s.mode_heldout_log_likelihood() for s in self.states_list)

    def M_step(self):
        self.M_step_init_dynamics_distn()
        super(LaplaceApproxPoissonLDS, self).M_step()

    def M_step_init_dynamics_distn(self):
        pass
        # TODO: the states object does not have a E_x1_x1 term...
        #       we could use the first of E_x_xT

        # self.init_dynamics_distn.max_likelihood(
        #     stats=(sum(s.E_x1_x1 for s in self.states_list)))

    def M_step_emission_distn(self):
        self.emission_distn.max_likelihood(
            stats=[s.E_emission_stats for s in self.states_list])

    def expected_log_likelihood(self):
        return sum([s.expected_log_likelihood() for s in self.states_list])

    # def initialize_with_pca(self, init_model=None, N_iter=100):
    #     from pglds.models import ApproxPoissonPCA
    #     from pybasicbayes.util.text import progprint_xrange
    #
    #     ### Initialize with PCA
    #     if init_model is None:
    #         init_model = ApproxPoissonPCA(self.N, self.D_latent)
    #
    #         for states in self.states_list:
    #             init_model.add_data(states.data.X, states.data.mask)
    #
    #         print("Initializing with PCA")
    #         [init_model.resample_model() for _ in progprint_xrange(N_iter)]
    #
    #     C0 = init_model.C
    #     b0 = init_model.emission_distn.b
    #
    #     self.init_dynamics_distn.mu = np.zeros(self.D_latent)
    #     self.init_dynamics_distn.sigma = np.eye(self.D_latent)
    #     self.emission_distn.C = C0.copy()
    #     self.emission_distn.b = b0.copy()
    #
    #     for states, pca_states in zip(self.states_list, init_model.states_list):
    #         states.stateseq = pca_states.gaussian_states.copy()
    #
    #     if hasattr(init_model, 'A'):
    #         self.dynamics_distn.A = init_model.A.copy()
    #         self.dynamics_distn.sigma = init_model.sigma_states.copy()
    #
    #         # self.resample_dynamics_distns()


##############################
#  convenience constructors  #
##############################

# TODO make data-dependent default constructors
def DefaultLDS(D_obs, D_latent, D_input=0,
               mu_init=None, sigma_init=None,
               A=None, B=None, sigma_states=None,
               C=None, D=None, sigma_obs=None):
    model = LDS(
        dynamics_distn=Regression(
            nu_0=D_latent + 1,
            S_0=D_latent * np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent + D_input)),
            K_0=D_latent * np.eye(D_latent + D_input)),
        emission_distn=Regression(
            nu_0=D_obs + 1,
            S_0=D_obs * np.eye(D_obs),
            M_0=np.zeros((D_obs, D_latent + D_input)),
            K_0=D_obs * np.eye(D_latent + D_input)))

    set_default = \
        lambda prm, val, default: \
            model.__setattr__(prm, val if val is not None else default)

    set_default("mu_init", mu_init, np.zeros(D_latent))
    set_default("sigma_init", sigma_init, np.eye(D_latent))

    set_default("A", A, 0.99 * random_rotation(D_latent))
    set_default("B", B, 0.1 * np.random.randn(D_latent, D_input))
    set_default("sigma_states", sigma_states, 0.1 * np.eye(D_latent))

    set_default("C", C, np.random.randn(D_obs, D_latent))
    set_default("D", D, 0.1 * np.random.randn(D_obs, D_input))
    set_default("sigma_obs", sigma_obs, 0.1 * np.eye(D_obs))

    return model

def DefaultPoissonLDS(D_obs, D_latent, D_input=0,
                      mu_init=None, sigma_init=None,
                      A=None, B=None, sigma_states=None,
                      C=None, d=None,
                      ):
    assert D_input == 0, "Inputs are not yet supported for Poisson LDS"
    model = LaplaceApproxPoissonLDS(
        init_dynamics_distn=
            Gaussian(mu_0=np.zeros(D_latent), sigma_0=np.eye(D_latent),
                     kappa_0=1.0, nu_0=D_latent + 1),
        dynamics_distn=
            Regression(A=0.9 * np.eye(D_latent), sigma=np.eye(D_latent),
                       nu_0=D_latent + 1, S_0=D_latent * np.eye(D_latent),
                       M_0=np.zeros((D_latent, D_latent)), K_0=D_latent * np.eye(D_latent)),
        emission_distn=
            PoissonRegression(D_obs, D_latent, verbose=False))

    set_default = \
        lambda prm, val, default: \
            model.__setattr__(prm, val if val is not None else default)

    set_default("mu_init", mu_init, np.zeros(D_latent))
    set_default("sigma_init", sigma_init, np.eye(D_latent))

    set_default("A", A, 0.99 * random_rotation(D_latent))
    set_default("B", B, 0.1 * np.random.randn(D_latent, D_input))
    set_default("sigma_states", sigma_states, 0.1 * np.eye(D_latent))

    set_default("C", C, np.random.randn(D_obs, D_latent))
    set_default("d", d, np.zeros((D_obs, 1)))

    return model
