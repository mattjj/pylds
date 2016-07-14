from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression, AutoRegression, DiagonalRegression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import LDS, DefaultLDS

npr.seed(0)


#########################
#  set some parameters  #
#########################
D_obs, D_latent = 1, 2
mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24), np.cos(np.pi/24)]])
sigma_states = 0.01*np.eye(2)

C = np.array([[10.,0.]])
sigma_obs = 0.01*np.eye(1)


###################
#  generate data  #
###################

truemodel = LDS(
    dynamics_distn=AutoRegression(
            A=A,sigma=sigma_states),
    emission_distn=DiagonalRegression(
            D_obs, D_latent, A=C, sigmasq=np.diag(sigma_obs)))

data, stateseq = truemodel.generate(2000)


###############
#  fit model  #
###############
model = LDS(
    dynamics_distn=AutoRegression(
            nu_0=D_latent+1,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent)),
    emission_distn=DiagonalRegression(D_obs, D_latent))
model.add_data(data)

def update(model):
    return model.meanfield_coordinate_descent_step()

for _ in progprint_xrange(100):
    model.resample_model()

vlbs = [update(model) for _ in progprint_xrange(100)]

plt.figure(figsize=(3,4))
plt.plot(vlbs)
plt.xlabel('iteration')
plt.ylabel('variational lower bound')


################
#  smoothing   #
################
E_C,_,_,_ = model.emission_distn.mf_expectations
smoothed_obs = model.states_list[0].smoothed_mus.dot(E_C.T)

################
#  predicting  #
################
Nseed = 1700
Npredict = 100
prediction_seed = data[:Nseed]

model.emission_distn.resample_from_mf()
predictions = model.sample_predictions(prediction_seed, Npredict)

plt.figure()
plt.plot(data, 'k')
plt.plot(smoothed_obs[:Nseed], ':k')
plt.plot(Nseed + np.arange(Npredict), predictions, 'b')
plt.xlabel('time index')
plt.ylabel('prediction')

plt.show()
