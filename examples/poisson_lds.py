from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from scipy.stats import poisson

from pybasicbayes.util.text import progprint_xrange
from pylds.models import DefaultPoissonLDS

npr.seed(0)

# Parameters
D_obs = 10
D_latent = 2
T = 5000

# True LDS Parameters
mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.01*np.eye(2)
C = np.random.randn(D_obs, D_latent)

# Simulate from a Poisson LDS
truemodel = DefaultPoissonLDS(D_obs, D_latent, A=A, sigma_states=sigma_states, C=C)
data, stateseq = truemodel.generate(T)

# Fit with a Poisson LDS
model = DefaultPoissonLDS(D_obs, D_latent)
model.add_data(data)
model.states_list[0].gaussian_states *= 0

N_iters = 50
def em_update(model):
    model.EM_step(verbose=True)
    ll = model.log_likelihood()
    return ll

lls = [em_update(model) for _ in progprint_xrange(N_iters)]

# Compute baseline likelihood under Poisson MLE model
rates = data.mean(0)
baseline = 0
for n in range(D_obs):
    baseline += poisson(rates[n]).logpmf(data[:,n]).sum()

# Plot the log likelihood over iterations
plt.plot(np.array(lls) / T / D_obs, '-', lw=2, label="model")
plt.plot([0, N_iters-1], baseline * np.ones(2) / T / D_obs, ':k', lw=2, label="baseline")
plt.xlabel('iteration')
plt.ylabel('log likelihood per datapoint')
plt.legend(loc="lower right")
plt.tight_layout()

# Plot the smoothed observations
fig = plt.figure(figsize=(6, 6))
smoothed_obs = model.states_list[0].smooth()
true_smoothed_obs = truemodel.states_list[0].smooth()

ylims = (-0.1, 1.1)
xlims = (0, min(T,1000))

n_subplots = min(D_obs, 6)
n_to_plot = np.arange(n_subplots)
for i,j in enumerate(n_to_plot):
    ax = fig.add_subplot(n_subplots,1,i+1)

    # Plot the inferred rate
    ax.plot([0], [0], 'ko', lw=2, label="observed data")
    ax.plot(true_smoothed_obs[:,j], 'k', lw=3, label="true mean")
    ax.plot(smoothed_obs[:,j], '--r', lw=2, label="inf mean")

    # Plot spike counts
    yl = ax.get_ylim()
    given_ts = np.where(data[:, j] == 1)[0]
    ax.plot(given_ts, (yl[1] * 1.05) * np.ones_like(given_ts), 'ko', markersize=5)

    if i == 0:
        plt.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.8))
    if i == n_subplots - 1:
        plt.xlabel('time index')

    ax.set_xlim(xlims)
    ax.set_ylim(yl[0], yl[1] * 1.1)
    ax.set_ylabel("$x_%d(t)$" % (j+1))

plt.tight_layout()
plt.show()
