from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression
from pybasicbayes.util.text import progprint_xrange
from pypolyagamma.distributions import BernoulliRegression
from pylds.models import CountLDS, DefaultBernoulliLDS

npr.seed(1)

# Parameters
D_obs = 10
D_latent = 2
D_input = 0
T = 2000

# True LDS Parameters
mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
B = np.zeros((D_latent, D_input))
sigma_states = 0.01*np.eye(2)

C = np.random.randn(D_obs, D_latent)
D = np.zeros((D_obs, D_input))
b = -2.0 * np.ones((D_obs, 1))

# Simulate from a Bernoulli LDS
truemodel = CountLDS(
    dynamics_distn=Regression(A=np.hstack((A,B)), sigma=sigma_states),
    emission_distn=BernoulliRegression(D_out=D_obs, D_in=D_latent + D_input,
                                       A=np.hstack((C,D)), b=b))

inputs = np.random.randn(T, D_input)
data, stateseq = truemodel.generate(T, inputs=inputs)

# Make a model
model = CountLDS(
    dynamics_distn=Regression(nu_0=D_latent + 2,
                              S_0=D_latent * np.eye(D_latent),
                              M_0=np.zeros((D_latent, D_latent + D_input)),
                              K_0=(D_latent + D_input) * np.eye(D_latent + D_input)),
    emission_distn=BernoulliRegression(D_out=D_obs, D_in=D_latent + D_input))
model.add_data(data, inputs=inputs, stateseq=np.zeros((T, D_latent)))

# Run a Gibbs sampler with Polya-gamma augmentation
N_samples = 50
def gibbs_update(model):
    model.resample_model()
    smoothed_obs = model.states_list[0].smooth()
    ll = model.log_likelihood()
    return ll, model.states_list[0].gaussian_states, smoothed_obs

lls_gibbs, x_smpls_gibbs, y_smooth_gibbs = \
    zip(*[gibbs_update(model) for _ in progprint_xrange(N_samples)])

# Fit with a Bernoulli LDS using Laplace approximation for comparison
model = DefaultBernoulliLDS(D_obs, D_latent, D_input=D_input,
                            C=0.01 * np.random.randn(D_obs, D_latent),
                            D=0.01 * np.random.randn(D_obs, D_input))
model.add_data(data, inputs=inputs, stateseq=np.zeros((T, D_latent)))

N_iters = 50
def em_update(model):
    model.EM_step(verbose=True)
    smoothed_obs = model.states_list[0].smooth()
    ll = model.log_likelihood()
    return ll, model.states_list[0].gaussian_states, smoothed_obs

lls_em, x_smpls_em, y_smooth_em = \
    zip(*[em_update(model) for _ in progprint_xrange(N_iters)])

# Plot the log likelihood over iterations
plt.figure(figsize=(10,6))
plt.plot(lls_gibbs, label="gibbs")
plt.plot(lls_em, label="em")
plt.plot([0,N_samples], truemodel.log_likelihood() * np.ones(2), '-k', label="true")
plt.xlabel('iteration')
plt.ylabel('log likelihood')
plt.legend(loc="lower right")

# Plot the smoothed observations
fig = plt.figure(figsize=(10,10))
N_subplots = min(D_obs, 6)

ylims = (-0.1, 1.1)
xlims = (0, min(T,1000))

n_to_plot = np.arange(min(N_subplots, D_obs))
for i,j in enumerate(n_to_plot):
    ax = fig.add_subplot(N_subplots,1,i+1)
    # Plot spike counts
    given_ts = np.where(data[:,j]==1)[0]
    ax.plot(given_ts, np.ones_like(given_ts), 'ko', markersize=5)

    ax.plot([0], [0], 'ko', lw=2, label="data")
    ax.plot(y_smooth_gibbs[-1][:, j], lw=2, label="gibbs probs")
    ax.plot(y_smooth_em[-1][:, j], lw=2, label="em probs")

    if i == 0:
        plt.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 2.))
    if i == N_subplots - 1:
        plt.xlabel('time index')
    ax.set_xlim(xlims)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("$x_%d(t)$" % (j+1))

plt.show()
