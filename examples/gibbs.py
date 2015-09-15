from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression
from pybasicbayes.util.text import progprint_xrange
from autoregressive.distributions import AutoRegression

from pylds.models import LDS, DefaultLDS

npr.seed(0)


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
    emission_distn=Regression(A=C,sigma=sigma_obs))

data, stateseq = truemodel.generate(2000)


###############
#  fit model  #
###############

def update(model):
    model.resample_model()
    return model.log_likelihood()

model = DefaultLDS(n=2,p=data.shape[1]).add_data(data)
vlbs = [update(model) for _ in progprint_xrange(100)]

plt.figure(figsize=(3,4))
plt.plot(vlbs)
plt.xlabel('iteration')
plt.ylabel('variational lower bound')


################
#  predicting  #
################

Npredict = 100
prediction_seed = data[:1700]

predictions = model.sample_predictions(
    prediction_seed, Npredict, obs_noise=False)

plt.figure()
plt.plot(data, 'b-')
plt.plot(prediction_seed.shape[0] + np.arange(Npredict), predictions, 'r--')
plt.xlabel('time index')
plt.ylabel('prediction')

plt.show()
