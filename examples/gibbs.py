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
input_model = DefaultLDS(D_obs, D_latent, D_input)
input_model.add_data(data, inputs=inputs)

# Fit a separate model without the inputs
noinput_model = DefaultLDS(D_obs, D_latent, D_input=0)
noinput_model.add_data(data)

# Run the Gibbs sampler
def update(model):
    model.resample_model()
    return model.log_likelihood()

input_lls = [update(input_model) for _ in progprint_xrange(100)]
noinput_lls = [update(noinput_model) for _ in progprint_xrange(100)]

# Plot the log likelihoods
plt.figure()
plt.plot(input_lls, label="with inputs")
plt.plot(noinput_lls, label="wo inputs")
plt.xlabel('iteration')
plt.ylabel('training likelihood')
plt.legend()

# Predict forward in time
T_given = 1800
T_predict = 200
given_data= data[:T_given]
given_inputs = inputs[:T_given]

preds = \
    input_model.sample_predictions(
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
input_ys = input_model.smooth(data, inputs)
noinput_ys = noinput_model.smooth(data)

plt.figure()
plt.plot(data, 'b-', label="true")
plt.plot(input_ys, 'r-', lw=2, label="with input")
plt.xlabel("Time")
plt.xlim(max(0, T_given-200), T)
plt.ylabel("Smoothed Data")
plt.legend()

plt.show()
