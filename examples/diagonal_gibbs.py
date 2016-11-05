from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression, DiagonalRegression
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
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.01*np.eye(2)

C = np.array([[10.,0.]])
sigma_obs = 0.01*np.eye(D_obs)


###################
#  generate data  #
###################

truemodel = LDS(
    dynamics_distn=Regression(A=A,sigma=sigma_states),
    emission_distn=DiagonalRegression(D_obs, D_latent, A=C, sigmasq=np.diag(sigma_obs)))

data, stateseq = truemodel.generate(2000)


###############
#  fit model  #
###############

def update(model):
    model.resample_model()
    return model.log_likelihood()

diag_model = LDS(
    dynamics_distn=Regression(
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent)),
    emission_distn=DiagonalRegression(D_obs, D_latent))

diag_model.add_data(data)

full_model = model = DefaultLDS(n=2,p=data.shape[1]).add_data(data)

diag_lls = [update(diag_model) for _ in progprint_xrange(200)]
full_lls = [update(full_model) for _ in progprint_xrange(200)]

plt.figure(figsize=(3,4))
plt.plot(diag_lls, label="diagonal")
plt.plot(full_lls, label="full")
plt.xlabel('iteration')
plt.ylabel('log likelihood')
plt.legend()

################
#  smoothing   #
################
smoothed_obs = diag_model.smooth(data)

################
#  predicting  #
################
Nseed = 1700
Npredict = 100
prediction_seed = data[:Nseed]

diag_preds = diag_model.sample_predictions(
    prediction_seed, Npredict, obs_noise=False)

full_preds = full_model.sample_predictions(
    prediction_seed, Npredict, obs_noise=False)

plt.figure()
plt.plot(data, 'k-')
plt.plot(smoothed_obs[:Nseed], ':k')
plt.plot(Nseed + np.arange(Npredict), diag_preds, 'b')
plt.plot(Nseed + np.arange(Npredict), full_preds, 'r')
plt.xlabel('time index')
plt.ylabel('prediction')
# plt.xlim(1800,2000)

plt.show()
