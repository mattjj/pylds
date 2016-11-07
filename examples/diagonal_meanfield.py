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

# Simulate from an LDS
truemodel = DefaultLDS(D_obs, D_latent, D_input)
inputs = np.random.randn(T, D_input)
data, stateseq = truemodel.generate(T, inputs=inputs)

# Fit with an LDS with diagonal observation noise
model = LDS(
    dynamics_distn=Regression(nu_0=D_latent + 2,
                              S_0=D_latent * np.eye(D_latent),
                              M_0=np.zeros((D_latent, D_latent + D_input)),
                              K_0=(D_latent + D_input) * np.eye(D_latent + D_input)),
    emission_distn=DiagonalRegression(D_obs, D_latent+D_input))
model.add_data(data, inputs=inputs)

# Fit with mean field
def update(model):
    return model.meanfield_coordinate_descent_step()

for _ in progprint_xrange(100):
    model.resample_model()

N_steps = 100
vlbs = [update(model) for _ in progprint_xrange(N_steps)]

plt.figure(figsize=(3,4))
plt.plot([0, N_steps], truemodel.log_likelihood() * np.ones(2), '--k')
plt.plot(vlbs)
plt.xlabel('iteration')
plt.ylabel('variational lower bound')

# Predict forward in time
T_given = 1800
T_predict = 200
given_data= data[:T_given]
given_inputs = inputs[:T_given]

preds = \
    model.sample_predictions(
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
E_CD,_,_,_ = model.emission_distn.mf_expectations
E_C, E_D = E_CD[:,:D_latent], E_CD[:,D_latent:]
ys = model.states_list[0].smoothed_mus.dot(E_C.T) + inputs.dot(E_D.T)

plt.figure()
plt.plot(data, 'b-', label="true")
plt.plot(ys, 'r-', lw=2, label="smoothed")
plt.xlabel("Time")
plt.xlim(max(0, T_given-200), T)
plt.ylabel("Smoothed Data")
plt.legend()

plt.show()
