from __future__ import division
import numpy as np

from pyhsmm.basic.abstractions import Model, ModelGibbsSampling, ModelEM

from states import LDSStates

# TODO make separate versions for stationary, nonstationary,
# nonstationary-and-distinct-for-each-sequence

# NOTE: dynamics_distn should probably be an instance of AutoRegression,
# emission_distn should probably be an instance of Regression, and
# init_dynamics_distn should probably be an instance of Gaussian

class _LDSBase(Model):
    def __init__(self,dynamics_distn,emission_distn,init_dynamics_distn):
        self.dynamics_distn = dynamics_distn
        self.emission_distn = emission_distn
        self.init_dynamics_distn = init_dynamics_distn
        self.states_list = []

    def add_data(self,data,**kwargs):
        self.states_list.append(LDSStates(model=self,data=data,**kwargs))

    def log_likelihood(self):
        raise NotImplementedError

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

    ### convenience properties

    @property
    def n(self):
        'latent dimension'
        return self.emission_distn.D_in

    @property
    def p(self):
        'emission dimension'
        return self.emission_distn.D_out

class _LDSGibbsSampling(_LDSBase,ModelGibbsSampling):
    def resample_model(self):
        self.resample_states()
        self.resample_parameters()

    def resample_states(self):
        for s in self.states_list:
            s.resample()

    def resample_parameters(self):
        self.resample_init_dynamics_distn()
        self.resample_dynamics_distn()
        self.resample_emission_distn()

    def resample_init_dynamics_distn(self):
        self.init_dynamics_distn.resample([s.stateseq[0] for s in self.states_list])

    def resample_dynamics_distn(self):
        self.dynamics_distn.resample([s.strided_stateseq for s in self.states_list])

    def resample_emission_distn(self):
        self.emission_distn.resample([np.hstack((s.stateseq,s.data)) for s in self.states_list])


class _LDSEM(_LDSBase,ModelEM):
    def EM_step(self):
        assert len(self.states_list) > 0
        self.E_step()
        self.M_step()

    def E_step(self):
        for s in self.states_list:
            s.E_step()

    def M_Step(self):
        self.M_step_init_dynamics_distn()
        self.M_step_dynamics_distn()
        self.M_step_emission_distn()

    def M_step_init_dynamics_distn(self):
        self.init_dynamics_distn.max_likelihood(
            stats=(sum(s.E_x1_x1 for s in self.states_list)))

    def M_step_dynamics_distn(self):
        self.dynamics_distn.max_likelihood(
            stats=(sum(s.E_xt_xtp1 for s in self.states_list)))

    def M_step_emission_distn(self):
        self.emission_distn.max_likelihood(
            stats=(sum(s.E_xt_yt for s in self.states_list)))

class LDS(_LDSGibbsSampling,_LDSEM):
    pass

