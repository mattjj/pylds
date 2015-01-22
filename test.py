from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Gaussian, Regression
from pybasicbayes.util.text import progprint_xrange
from autoregressive.distributions import AutoRegression

from models import LDS

np.random.seed(0)

#########################
#  set some parameters  #
#########################

mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.01*np.eye(2)

C = np.array([[10.,0.]])
sigma_obs = 0.01*np.eye(1)

###################
#  generate data  #
###################

truemodel = LDS(
        dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
        emission_distn=Regression(A=C,sigma=sigma_obs)
        )

data, stateseq = truemodel.generate(2000)

###############
#  fit model  #
###############

model = LDS(
        dynamics_distn=AutoRegression(nu_0=3.,S_0=np.eye(2),M_0=np.zeros((2,2)),K_0=5*np.eye(2)),
        emission_distn=Regression(nu_0=2.,S_0=np.eye(1),M_0=np.zeros((1,2)),K_0=5*np.eye(2)),
        )

model.add_data(data)

model.resample_parameters()
for _ in progprint_xrange(100):
    model.resample_model()

print np.linalg.eigvals(A)
print np.linalg.eigvals(model.dynamics_distn.A)

