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
D_input = 0
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
data, stateseq = truemodel.generate(T, inputs=inputs)


###############
# make model  #
###############
model = LDS(
    dynamics_distn=Regression(nu_0=D_latent + 2, S_0=D_latent * np.eye(D_latent),
                              M_0=np.zeros((D_latent, D_latent + D_input)), K_0=(D_latent + D_input) * np.eye(D_latent + D_input)),
    emission_distn=Regression(nu_0=D_obs + 1, S_0=D_obs * np.eye(D_obs),
                              M_0=np.zeros((D_obs, D_latent + D_input)), K_0=(D_latent + D_input) * np.eye(D_latent + D_input)))
model.add_data(data, inputs=inputs)

###############
#  fit model  #
###############
def update(model):
    return model.meanfield_coordinate_descent_step()

for _ in progprint_xrange(100):
    model.resample_model()

vlbs = [update(model) for _ in progprint_xrange(500)]
model.resample_from_mf()

plt.figure(figsize=(3,4))
plt.plot(vlbs)
plt.xlabel('iteration')
plt.ylabel('variational lower bound')
plt.show()

################
#  predicting  #
################
Nseed = 1700
Npredict = 100
given_data = data[:Nseed]
given_inputs = inputs[:Nseed]
pred_inputs = inputs[Nseed:Nseed+Npredict]
predictions = model.sample_predictions(given_data,
                                       inputs=given_inputs,
                                       Tpred=Npredict,
                                       inputs_pred=pred_inputs)

plt.figure()
plt.plot(data, 'b-')
plt.plot(given_data.shape[0] + np.arange(Npredict), predictions, 'r--')
plt.xlabel('time index')
plt.ylabel('prediction')

plt.show()
