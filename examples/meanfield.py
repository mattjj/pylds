from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import LDS

npr.seed(0)


#########################
#  set some parameters  #
#########################
D_obs = 1
D_latent = 2
D_input = 1
T = 2000

mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
B = np.ones((D_latent, D_input))
sigma_states = 0.01*np.eye(2)

C = np.array([[10.,0.]])
D = np.zeros((D_obs, D_input))
sigma_obs = 0.01*np.eye(1)

###################
#  generate data  #
###################

truemodel = LDS(
    dynamics_distn=Regression(A=np.hstack((A,B)), sigma=sigma_states),
    emission_distn=Regression(A=np.hstack((C,D)), sigma=sigma_obs))

inputs = np.random.randn(T, D_input)
# inputs = np.zeros((T, D_input))
data, stateseq = truemodel.generate(T, inputs=inputs)


###############
# make model  #
###############
model = LDS(
    dynamics_distn=Regression(nu_0=D_latent + 2,
                              S_0=D_latent * np.eye(D_latent),
                              M_0=np.zeros((D_latent, D_latent + D_input)),
                              K_0=(D_latent + D_input) * np.eye(D_latent + D_input),
                              # A=np.hstack((A,B)), sigma=sigma_states
                              ),
    emission_distn=Regression(nu_0=D_obs + 2,
                              S_0=D_obs * np.eye(D_obs),
                              M_0=np.zeros((D_obs, D_latent + D_input)),
                              K_0=(D_latent + D_input) * np.eye(D_latent + D_input),
                              # A=np.hstack((C,D)), sigma=100*sigma_obs
                              )
    )
model.add_data(data, inputs=inputs)
# model.emission_distn._initialize_mean_field()
# model.dynamics_distn._initialize_mean_field()

###############
#  fit model  #
###############
def update(model):
    return model.meanfield_coordinate_descent_step()

for _ in progprint_xrange(100):
    model.resample_model()

N_steps = 100
vlbs = [update(model) for _ in progprint_xrange(N_steps)]
model.resample_from_mf()

plt.figure(figsize=(3,4))
plt.plot([0, N_steps], truemodel.log_likelihood()*np.ones(2), '--k')
plt.plot(vlbs)
plt.xlabel('iteration')
plt.ylabel('variational lower bound')
plt.show()

################
#  smoothing   #
################
smoothed_obs = model.states_list[0].meanfield_smooth()

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
