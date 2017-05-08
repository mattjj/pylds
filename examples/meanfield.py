from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.util.text import progprint_xrange

from pylds.models import DefaultLDS

npr.seed(0)

# Set parameters
D_obs = 1
D_latent = 2
D_input = 1
T = 2000

# Simulate from one LDS
truemodel = DefaultLDS(D_obs, D_latent, D_input)
inputs = np.random.randn(T, D_input)
data, stateseq = truemodel.generate(T, inputs=inputs)

# Fit with another LDS
model = DefaultLDS(D_obs, D_latent, D_input)
model.add_data(data, inputs=inputs)

# Initialize with a few iterations of Gibbs
for _ in progprint_xrange(10):
    model.resample_model()

# Run EM
def update(model):
    vlb = model.meanfield_coordinate_descent_step()
    return vlb

vlbs = [update(model) for _ in progprint_xrange(50)]

# Sample from the mean field posterior
model.resample_from_mf()

# Plot the log likelihoods
plt.figure()
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


# Smooth the data
ys = model.smooth(data, inputs)

plt.figure()
plt.plot(data, 'b-', label="true")
plt.plot(ys, 'r-', lw=2, label="smoothed")
plt.xlabel("Time")
plt.xlim(max(0, T_given-200), T)
plt.ylabel("Smoothed Data")
plt.legend()

plt.show()
