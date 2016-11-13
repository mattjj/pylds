from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression
from pybasicbayes.util.text import progprint_xrange
from pypolyagamma.distributions import BernoulliRegression
from pylds.models import LDS, DefaultPoissonLDS

npr.seed(0)

# Parameters
D_obs = 10
D_latent = 2
T = 2000

# True LDS Parameters
mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.01*np.eye(2)

C = np.random.randn(D_obs, D_latent)
b = -2.0 * np.ones((D_obs, 1))

# Simulate from a Bernoulli LDS
truemodel = LDS(
    dynamics_distn=Regression(A=A, sigma=sigma_states),
    emission_distn=BernoulliRegression(D_out=D_obs, D_in=D_latent, A=C, b=b))
data, stateseq = truemodel.generate(T)

# Fit with a Poisson LDS
model = DefaultPoissonLDS(D_obs, D_latent)
model.add_data(data, verbose=False)

N_iters = 50
def em_update(model):
    model.EM_step()
    ll = model.log_likelihood()
    return ll

lls = [em_update(model) for _ in progprint_xrange(N_iters)]

# Plot the log likelihood over iterations
plt.figure(figsize=(10,6))
plt.plot(lls,'-b')
plt.xlabel('iteration')
plt.ylabel('log likelihood')

# Plot the smoothed observations
fig = plt.figure(figsize=(10,10))
N_subplots = min(D_obs, 6)
smoothed_obs = model.states_list[0].smooth()
true_smoothed_obs = truemodel.states_list[0].smooth()

ylims = (-0.1, 1.1)
xlims = (0, min(T,1000))

n_to_plot = np.arange(min(N_subplots, D_obs))
for i,j in enumerate(n_to_plot):
    ax = fig.add_subplot(N_subplots,1,i+1)
    # Plot spike counts
    given_ts = np.where(data[:,j]==1)[0]
    ax.plot(given_ts, np.ones_like(given_ts), 'ko', markersize=5)

    # Plot the inferred rate
    ax.plot([0], [0], 'ko', lw=2, label="observed data")
    ax.plot(smoothed_obs[:,j], 'r', lw=2, label="poisson mean")
    ax.plot(true_smoothed_obs[:,j], 'k', lw=2, label="true mean")

    if i == 0:
        plt.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.8))
    if i == N_subplots - 1:
        plt.xlabel('time index')
    ax.set_xlim(xlims)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("$x_%d(t)$" % (j+1))

plt.show()

