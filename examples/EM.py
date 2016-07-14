from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import LDS, DefaultLDS

npr.seed(0)


#########################
#  set some parameters  #
#########################

p = 1
n = 2
d = 1
T = 2000

mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
B = np.ones((n,d))
sigma_states = 0.01*np.eye(2)

C = np.array([[10.,0.]])
D = np.zeros((p,d))
sigma_obs = 0.01*np.eye(1)

###################
#  generate data  #
###################

truemodel = LDS(
    dynamics_distn=Regression(A=np.hstack((A,B)), sigma=sigma_states),
    emission_distn=Regression(A=np.hstack((C,D)), sigma=sigma_obs))

inputs = np.random.randn(T,d)
data, stateseq = truemodel.generate(T, inputs=inputs)

###############
# test models #
###############
model = LDS(
    dynamics_distn=Regression(nu_0=n + 2, S_0=n * np.eye(n),
                              M_0=np.zeros((n, n+d)), K_0=(n+d) * np.eye(n+d)),
    emission_distn=Regression(nu_0=p+1, S_0=p*np.eye(p),
                              M_0=np.zeros((p, n+d)), K_0=(n+d)*np.eye(n+d)))
model.add_data(data, inputs=inputs)

###############
#  fit model  #
###############

def update(model):
    model.EM_step()
    return model.log_likelihood()


for _ in progprint_xrange(10):
    model.resample_model()

likes = [update(model) for _ in progprint_xrange(50)]

plt.figure(figsize=(3,4))
plt.plot(likes)
plt.xlabel('iteration')
plt.ylabel('training likelihood')


################
#  predicting  #
################

Nseed = 1700
Npredict = 100
prediction_seed = data[:Nseed]

preds = \
    model.sample_predictions(
        prediction_seed, Npredict,
        inputs=inputs[Nseed:Nseed+Npredict])

plt.figure()
plt.plot(data, 'b-')
plt.plot(Nseed + np.arange(Npredict), preds, 'r--')
# plt.plot(Nseed + np.arange(Npredict), noinput_preds, 'g--')
plt.xlabel('time index')
plt.ylabel('prediction')

plt.show()

