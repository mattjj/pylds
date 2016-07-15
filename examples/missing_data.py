from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import AutoRegression, DiagonalRegression
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
    dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
    emission_distn=DiagonalRegression(D_obs, D_latent, A=C, sigmasq=np.diag(sigma_obs)))

T = 2000
data, stateseq = truemodel.generate(T)

# Mask off a chunk of data
mask = np.ones_like(data, dtype=bool)
mask_start = 900
mask_stop = 1200
mask_len = mask_stop - mask_start
mask[mask_start:mask_stop] = False

###############
#  fit model  #
###############

def update(model):
    model.resample_model()
    return model.log_likelihood()

model = LDS(
    dynamics_distn=AutoRegression(
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent)),
    emission_distn=DiagonalRegression(D_obs, D_latent))
model.add_data(data=data, mask=mask)

lls = [update(model) for _ in progprint_xrange(100)]

plt.figure(figsize=(3,4))
plt.plot(lls)
plt.xlabel('iteration')
plt.ylabel('log likelihood')


################
#  smoothing   #
################
smoothed_obs = model.smooth(data, mask=mask)

plt.figure()
given_data = data.copy()
given_data[~mask] = np.nan
masked_data = data.copy()
masked_data[mask] = np.nan

plt.plot(given_data, 'k', label="observed")
plt.plot(masked_data, ':k', label="masked")
plt.plot(smoothed_obs, 'b', label="smoothed")
plt.xlabel('time index')
plt.ylabel('observation')
plt.xlim([max(mask_start-mask_len, 0), min(mask_stop+mask_len, T)])
plt.legend(loc="upper right")

plt.show()

################
#  predicting  #
################
Nseed = 1700
Npredict = 100
prediction_seed = data[:Nseed]

preds = model.sample_predictions(
    prediction_seed, Npredict, obs_noise=False)

plt.figure()
plt.plot(given_data, 'k', label="observed")
plt.plot(masked_data, ':k', label="masked")
plt.plot(Nseed + np.arange(Npredict), preds, 'b')
plt.xlabel('time index')
plt.ylabel('prediction')
# plt.xlim(1800,2000)

plt.show()

