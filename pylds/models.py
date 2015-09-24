from __future__ import division
import numpy as np

from pybasicbayes.abstractions import Model, ModelGibbsSampling, \
    ModelEM, ModelMeanField, ModelMeanFieldSVI

from states import LDSStates

# TODO make separate versions for stationary, nonstationary,
# nonstationary-and-distinct-for-each-sequence

# NOTE: dynamics_distn should probably be an instance of AutoRegression,
# emission_distn should probably be an instance of Regression, and
# init_dynamics_distn should probably be an instance of Gaussian


######################
#  algorithm mixins  #
######################

class _LDSBase(Model):
    def __init__(self,dynamics_distn,emission_distn):
        self.dynamics_distn = dynamics_distn
        self.emission_distn = emission_distn
        self.states_list = []

    def add_data(self,data,**kwargs):
        assert isinstance(data,np.ndarray)
        self.states_list.append(LDSStates(model=self,data=data,**kwargs))
        return self

    def log_likelihood(self,data=None):
        if data is not None:
            assert isinstance(data,(list,np.ndarray))
            if isinstance(data,np.ndarray):
                self.add_data(data=data,generate=False)
                return self.states_list.pop().log_likelihood()
            else:
                return sum(self.log_likelihood(d) for d in data)
        else:
            return sum(s.log_likelihood() for s in self.states_list)

    def generate(self,T,keep=True):
        s = LDSStates(model=self,T=T,initialize_from_prior=True)
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return data, s.stateseq

    def _generate_obs(self,s):
        if s.data is None:
            s.data = self.emission_distn.rvs(x=s.stateseq,return_xy=False)
        else:
            # filling in missing data
            raise NotImplementedError
        return s.data

    def predict(self, data, Tpred):
        # return means and covariances
        raise NotImplementedError

    def sample_predictions(self, data, Tpred, states_noise=True, obs_noise=True):
        self.add_data(data, generate=False)
        s = self.states_list.pop()
        return s.sample_predictions(Tpred, states_noise, obs_noise)

    # convenience properties

    @property
    def n(self):
        'latent dimension'
        return self.emission_distn.D_in

    @property
    def p(self):
        'emission dimension'
        return self.emission_distn.D_out

    @property
    def mu_init(self):
        return np.zeros(self.n) if not hasattr(self,'_mu_init') \
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
                np.eye(self.n**2) - np.kron(self.A,self.A), self.sigma_states.ravel())\
                .reshape(self.n,self.n)

    @sigma_init.setter
    def sigma_init(self,sigma_init):
        self._sigma_init = sigma_init

    @property
    def A(self):
        return self.dynamics_distn.A

    @A.setter
    def A(self,A):
        self.dynamics_distn.A = A

    @property
    def sigma_states(self):
        return self.dynamics_distn.sigma

    @sigma_states.setter
    def sigma_states(self,sigma_states):
        self.dynamics_distn.sigma = sigma_states

    @property
    def C(self):
        return self.emission_distn.A

    @C.setter
    def C(self,C):
        self.emission_distn.A = C

    @property
    def sigma_obs(self):
        return self.emission_distn.sigma

    @sigma_obs.setter
    def sigma_obs(self,sigma_obs):
        self.emission_distn.sigma = sigma_obs

    @property
    def is_stable(self):
        return np.max(np.abs(np.linalg.eivals(self.dynamics_distn.A))) < 1.


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
            [s.strided_stateseq for s in self.states_list])

    def resample_emission_distn(self):
        self.emission_distn.resample(
            [np.hstack((s.stateseq,s.data)) for s in self.states_list])


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

    def vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += self.emission_distn.get_vlb()
        vlb += self.dynamics_distn.get_vlb()
        return vlb


class _LDSMeanFieldSVI(_LDSBase, ModelMeanFieldSVI):
    def meanfield_sgdstep(self, minibatch, prob, stepsize, **kwargs):
        states_list = self._get_mb_states_list(minibatch, **kwargs)
        for s in states_list:
            s.meanfieldupdate()
        self._meanfield_sgdstep_parameters(states_list, prob, stepsize)

    def _meanfield_sgdstep_parameters(self, states_list, prob, stepsize):
        self._meanfield_sgdstep_dynamics_distn(states_list, prob, stepsize)
        self._meanfield_sgdstep_emission_distn(states_list, prob, stepsize)

    def _meanfield_sgdstep_dynamics_distn(self, states_list, prob, stepsize):
        self.dynamics_distn.meanfield_sgdstep(
            stats=(sum(s.E_dynamics_stats for s in states_list)),
            prob=prob, stepsize=stepsize)

    def _meanfield_sgdstep_emission_distn(self, states_list, prob, stepsize):
        self.emission_distn.meanfield_sgdstep(
            stats=(sum(s.E_emission_stats for s in states_list)),
            prob=prob, stepsize=stepsize)

    def _get_mb_states_list(self, minibatch, **kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]

        def get_states(data):
            self.add_states(data, generate=False, **kwargs)
            return self.states_list.pop()

        return [get_states(data) for data in minibatch]


class _NonstationaryLDSGibbsSampling(_LDSGibbsSampling):
    def resample_model(self):
        self.resample_init_dynamics_distn()
        super(_NonstationaryLDSGibbsSampling, self).resample_model()

    def resample_init_dynamics_distn(self):
        self.init_dynamics_distn.resample(
            [s.stateseq[0] for s in self.states_list])


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

class LDS(_LDSGibbsSampling, _LDSMeanField, _LDSEM, _LDSBase):
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
            [s.stateseq[0] for s in self.states_list])

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


##############################
#  convenience constructors  #
##############################

# TODO make data-dependent default constructors
# TODO make a constructor that takes A, B, C, D

from pybasicbayes.distributions import Regression
from autoregressive.distributions import AutoRegression


def DefaultLDS(n, p):
    model = LDS(
        dynamics_distn=AutoRegression(
            nu_0=n+1, S_0=n*np.eye(n), M_0=np.zeros((n, n)), K_0=n*np.eye(n)),
        emission_distn=Regression(
            nu_0=p+1, S_0=p*np.eye(p), M_0=np.zeros((p, n)), K_0=p*np.eye(n)))

    model.A = 0.99*np.eye(n)
    model.sigma_states = np.eye(n)
    model.C = np.random.randn(p, n)
    model.sigma_obs = 0.1*np.eye(p)

    return model
