from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import AutoRegression, DiagonalRegression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import LDS

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

# C = np.array([[10.,0.]])
C = np.random.randn(D_obs, D_latent)
sigma_obs = 0.1*np.eye(D_obs)


###################
#  generate data  #
###################

truemodel = LDS(
    dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
    emission_distn=DiagonalRegression(D_obs, D_latent, A=C, sigmasq=np.diag(sigma_obs)))

truemodel.mu_init = mu_init
truemodel.sigma_init = sigma_init

T = 2000
data, stateseq = truemodel.generate(T)

# Mask off a chunk of data
mask = np.ones_like(data, dtype=bool)
mask_start = 1000
mask_stop = 1100
mask_len = mask_stop - mask_start
mask[mask_start:mask_stop] = False

true_ll = truemodel.log_likelihood()

###############
#  make model #
###############
model = LDS(
    dynamics_distn=AutoRegression(
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent)),
    emission_distn=DiagonalRegression(D_obs, D_latent, alpha_0=2.0, beta_0=1.0))
model.add_data(data=data, mask=mask)

print("True LL: ", truemodel.log_likelihood(data, mask))
print("Init LL: ", model.log_likelihood(data, mask))


###############
#  fit model  #
###############
N_samples = 500
sigma_obs_smpls = []
def gibbs_update(model):
    model.resample_model()
    sigma_obs_smpls.append(model.sigma_obs_flat[0])
    return model.log_likelihood()

def em_update(model):
    model.EM_step()
    sigma_obs_smpls.append(model.sigma_obs_flat[0])
    return model.log_likelihood()

def meanfield_update(model):
    model.meanfield_coordinate_descent_step()
    sigma_obs_smpls.append(model.emission_distn.mf_beta / model.emission_distn.mf_alpha)
    return model.log_likelihood()

# Gibbs
# lls = [gibbs_update(model) for _ in progprint_xrange(N_samples)]

# EM -- initialized with a few Gibbs iterations
[model.resample_model() for _ in progprint_xrange(100)]
lls = [em_update(model) for _ in progprint_xrange(N_samples)]

# Mean field
# lls = [meanfield_update(model) for _ in progprint_xrange(N_samples)]

plt.figure()
plt.plot(sigma_obs_smpls)
plt.xlabel("iteration")
plt.ylabel("sigma_obs")

plt.figure()
plt.plot(lls,'-b')
plt.plot([0,N_samples], truemodel.log_likelihood(data, mask) * np.ones(2), '-k')
plt.xlabel('iteration')
plt.ylabel('log likelihood')



################
#  smoothing   #
################
smoothed_obs = model.smooth(data, mask=mask)
sample_predictive_obs = model.states_list[0].stateseq.dot(model.C.T)

plt.figure()
given_data = data.copy()
given_data[~mask] = np.nan
masked_data = data.copy()
masked_data[mask] = np.nan

plt.plot(given_data, 'k', label="observed")
plt.plot(masked_data, ':k', label="masked")
plt.plot(smoothed_obs, 'b', label="smoothed")
plt.plot(sample_predictive_obs, ':b', label="sample")
plt.xlabel('time index')
plt.ylabel('observation')
plt.xlim([max(mask_start-mask_len, 0), min(mask_stop+mask_len, T)])
plt.legend(loc="upper right")


################
#  predicting  #
################
# Nseed = 1700
# Npredict = 100
# prediction_seed = data[:Nseed]
#
# preds = model.sample_predictions(
#     prediction_seed, Npredict, obs_noise=False)
#
# plt.figure()
# plt.plot(given_data, 'k', label="observed")
# plt.plot(masked_data, ':k', label="masked")
# plt.plot(Nseed + np.arange(Npredict), preds, 'b')
# plt.xlabel('time index')
# plt.ylabel('prediction')

plt.show()

