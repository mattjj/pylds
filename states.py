from __future__ import division
import numpy as np

from pyhsmm.util.general import AR_striding

from lds_messages import filter_and_sample

# TODO stable initialization

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
    def init_dynamics_distn(self):
        return self.model.init_dynamics_distn

    @property
    def n(self):
        return self.model.n

    @property
    def p(self):
        return self.model.p

    @property
    def mu_init(self):
        return self.init_dynamics_distn.mu

    @property
    def sigma_init(self):
        return self.init_dynamics_distn.sigma

    @property
    def A(self):
        return self.dynamics_distn.A

    @property
    def sigma_states(self):
        return self.dynamics_distn.sigma

    @property
    def C(self):
        return self.emission_distn.A

    @property
    def sigma_obs(self):
        return self.emission_distn.sigma

    @property
    def strided_stateseq(self):
        return AR_striding(self.stateseq,1)

    ### generation

    def generate_states(self):
        # TODO make a cython version
        T, n, p = self.T, self.n, self.p

        stateseq = self.stateseq = np.empty((T,n),dtype='double')
        stateseq[0] = np.random.multivariate_normal(self.mu_init, self.sigma_init)
        for t in xrange(1,T):
            stateseq[t] = np.random.multivariate_normal(self.A.dot(stateseq[t-1]), self.sigma_states)

        return stateseq

    ### resampling

    def resample(self):
        self.stateseq = filter_and_sample(self.mu_init, self.sigma_init,
                self.A, self.sigma_states, self.C, self.sigma_obs,
                self.data)

    def E_step(self):
        raise NotImplementedError # TODO kalman smoother

