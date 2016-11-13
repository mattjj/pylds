# from __future__ import division
# import numpy as np
# import numpy.random as npr
# import matplotlib.pyplot as plt
#
# from pybasicbayes.distributions import Regression, DiagonalRegression
# from pybasicbayes.util.text import progprint_xrange
#
# from pylds.models import LDS, DefaultLDS
#
# npr.seed(0)
#
#
# #########################
# #  set some parameters  #
# #########################
# D_obs, D_latent = 1, 2
# mu_init = np.array([0.,1.])
# sigma_init = 0.01*np.eye(2)
#
# A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
#                    [np.sin(np.pi/24),  np.cos(np.pi/24)]])
# sigma_states = 0.01*np.eye(2)
#
# C = np.array([[10.,0.]])
# sigma_obs = 0.01*np.eye(D_obs)
#
#
# ###################
# #  generate data  #
# ###################
#
# truemodel = LDS(
#     dynamics_distn=Regression(A=A,sigma=sigma_states),
#     emission_distn=DiagonalRegression(D_obs, D_latent, A=C, sigmasq=np.diag(sigma_obs)))
#
# data, stateseq = truemodel.generate(2000)
#
#
# ###############
# #  fit model  #
# ###############
#
# def update(model):
#     model.resample_model()
#     return model.log_likelihood()
#
# diag_model = LDS(
#     dynamics_distn=Regression(
#             nu_0=D_latent+3,
#             S_0=D_latent*np.eye(D_latent),
#             M_0=np.zeros((D_latent, D_latent)),
#             K_0=D_latent*np.eye(D_latent)),
#     emission_distn=DiagonalRegression(D_obs, D_latent))
#
# diag_model.add_data(data)
#
# full_model = model = DefaultLDS(D_latent=2, D_obs=data.shape[1]).add_data(data)
#
# diag_lls = [update(diag_model) for _ in progprint_xrange(200)]
# full_lls = [update(full_model) for _ in progprint_xrange(200)]
#
# plt.figure(figsize=(3,4))
# plt.plot(diag_lls, label="diagonal")
# plt.plot(full_lls, label="full")
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.legend()
#
# ################
# #  smoothing   #
# ################
# smoothed_obs = diag_model.smooth(data)
#
# ################
# #  predicting  #
# ################
# Nseed = 1700
# Npredict = 100
# prediction_seed = data[:Nseed]
#
# diag_preds = diag_model.sample_predictions(
#     prediction_seed, Npredict, obs_noise=False)
#
# full_preds = full_model.sample_predictions(
#     prediction_seed, Npredict, obs_noise=False)
#
# plt.figure()
# plt.plot(data, 'k-')
# plt.plot(smoothed_obs[:Nseed], ':k')
# plt.plot(Nseed + np.arange(Npredict), diag_preds, 'b')
# plt.plot(Nseed + np.arange(Npredict), full_preds, 'r')
# plt.xlabel('time index')
# plt.ylabel('prediction')
# # plt.xlim(1800,2000)
#
# plt.show()
#

from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression, DiagonalRegression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import LDS, DefaultLDS

npr.seed(0)


# Parameters
D_obs = 1
D_latent = 2
D_input = 0
T = 2000

# Simulate from an LDS with diagonal observation noise
truemodel = DefaultLDS(D_obs, D_latent, D_input, sigma_obs=0.1 * np.eye(D_obs))
inputs = np.random.randn(T, D_input)
data, stateseq = truemodel.generate(T, inputs=inputs)

# Fit with an LDS with diagonal observation noise
diag_model = LDS(
    dynamics_distn=Regression(nu_0=D_latent + 2,
                              S_0=D_latent * np.eye(D_latent),
                              M_0=np.zeros((D_latent, D_latent + D_input)),
                              K_0=(D_latent + D_input) * np.eye(D_latent + D_input)),
    emission_distn=DiagonalRegression(D_obs, D_latent+D_input))
diag_model.add_data(data, inputs=inputs)

# Also fit a model with a full covariance matrix
full_model = DefaultLDS(D_obs, D_latent, D_input)
full_model.add_data(data, inputs=inputs)

# Fit with Gibbs sampling
def update(model):
    model.resample_model()
    return model.log_likelihood()

N_steps = 100
diag_lls = [update(diag_model) for _ in progprint_xrange(N_steps)]
full_lls = [update(full_model) for _ in progprint_xrange(N_steps)]

plt.figure()
plt.plot([0, N_steps], truemodel.log_likelihood() * np.ones(2), '--k', label="true")
plt.plot(diag_lls, label="diag cov.")
plt.plot(full_lls, label="full cov.")
plt.xlabel('iteration')
plt.ylabel('log likelihood')
plt.legend()

# Predict forward in time
T_given = 1800
T_predict = 200
given_data= data[:T_given]
given_inputs = inputs[:T_given]

preds = \
    diag_model.sample_predictions(
        given_data, inputs=given_inputs,
        Tpred=T_predict,
        inputs_pred=inputs[T_given:T_given + T_predict])

# Plot the predictions
plt.figure()
plt.plot(np.arange(T), data, 'b-', label="true")
plt.plot(T_given + np.arange(T_predict), preds, 'r--', label="prediction")
ylim = plt.ylim()
plt.plot([T_given, T_given], ylim, '-k')
plt.xlabel('time index')
plt.xlim(max(0, T_given - 200), T)
plt.ylabel('prediction')
plt.ylim(ylim)
plt.legend()

# Smooth the data (TODO: Clean this up)
ys = diag_model.smooth(data, inputs)

plt.figure()
plt.plot(data, 'b-', label="true")
plt.plot(ys, 'r-', lw=2, label="smoothed")
plt.xlabel("Time")
plt.xlim(max(0, T_given-200), T)
plt.ylabel("Smoothed Data")
plt.legend()

plt.show()
