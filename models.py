from __future__ import division
import numpy as np

from pybasicbayes.abstractions import Model, ModelGibbsSampling, ModelEM

from states import LDSStates

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
        self.states_list.append(LDSStates(data,**kwargs))

    def generate(self):
        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError

    ### convenience properties

    @property
    def n(self):
        return self.emission_distn.D_in

    @property
    def p(self):
        return self.emission_distn.D_out

class _LDSGibbsSampling(ModelGibbsSampling):
    def resample(self):
        self.resample_states()
        self.resample_parameters()

    def resample_states(self):
        for s in self.states_list:
            s.resample()

    def resample_parameters(self):
        self.resample_init_dynamics_distn()
        self.resample_dynamcis_distn()
        self.resample_emission_distn()

    def resample_init_dynamics_distn(self):
        self.init_dynamics_distn.resample([s.stateseq[0] for s in self.states_list])

    def resample_dynamics_distn(self):
        self.dynamics_distn.resample([s.strided_stateseq for s in self.states_list])

    def resample_emission_distn(self):
        self.emission_distn.resample([np.hstack((s.stateseq,s.data)) for s in self.states_list])


def _LDSEM(ModelEM):
    def EM_step(self):
        assert len(self.states_list) > 0
        self._E_step()
        self._M_step()

    def _E_step(self):
        raise NotImplementedError

    def _M_Step(self):
        raise NotImplementedError

class LDS(_LDSGibbsSampling,_LDSEM):
    pass

