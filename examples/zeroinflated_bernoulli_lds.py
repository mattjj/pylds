from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
# Fancy plotting
try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue",
                   "red",
                   "amber",
                   "faded green",
                   "dusty purple",
                   "crimson",
                   "greyish"]
    colors = sns.xkcd_palette(color_names)
except:
    colors = ['b' ,'r', 'y', 'g']


from pybasicbayes.distributions import Regression
from pybasicbayes.util.text import progprint_xrange
from pypolyagamma.distributions import BernoulliRegression
from pylds.models import ZeroInflatedCountLDS, LDS

npr.seed(0)

# Parameters
rho = 0.5    # Sparsity (1-probability of deterministic zero)
D_obs = 10
D_latent = 2
D_input = 0
T = 2000

### True LDS Parameters
mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
B = np.ones((D_latent, D_input))
sigma_states = 0.01*np.eye(2)

C = np.random.randn(D_obs, D_latent)
D = np.zeros((D_obs, D_input))
b = -2.0 * np.ones((D_obs, 1))

### Simulate from a Bernoulli LDS
truemodel = ZeroInflatedCountLDS(
    rho=rho,
    dynamics_distn=Regression(A=np.hstack((A,B)), sigma=sigma_states),
    emission_distn=BernoulliRegression(D_out=D_obs, D_in=D_latent + D_input,
                                       A=np.hstack((C,D)), b=b))

inputs = np.random.randn(T, D_input)
data, stateseq = truemodel.generate(T, inputs=inputs)
dense_data = data.toarray()
true_rate = rho * truemodel.emission_distn.mean(np.hstack((stateseq, inputs)))

### First fit a zero inflated model
zi_model = ZeroInflatedCountLDS(
    rho=rho,
    dynamics_distn=Regression(nu_0=D_latent + 2,
                              S_0=D_latent * np.eye(D_latent),
                              M_0=np.zeros((D_latent, D_latent + D_input)),
                              K_0=(D_latent + D_input) * np.eye(D_latent + D_input)),
    emission_distn=BernoulliRegression(D_out=D_obs, D_in=D_latent + D_input))
zi_model.add_data(data, inputs=inputs)

# Run a Gibbs sampler
N_samples = 500
def gibbs_update(model):
    model.resample_model()
    return model.log_likelihood(), \
           model.states_list[0].gaussian_states, \
           model.states_list[0].smooth()

zi_lls, zi_x_smpls, zi_smoothed_obss = \
    zip(*[gibbs_update(zi_model) for _ in progprint_xrange(N_samples)])

### Now fit a standard model
std_model = LDS(
    dynamics_distn=Regression(nu_0=D_latent + 2,
                              S_0=D_latent * np.eye(D_latent),
                              M_0=np.zeros((D_latent, D_latent + D_input)),
                              K_0=(D_latent + D_input) * np.eye(D_latent + D_input)),
    emission_distn=BernoulliRegression(D_out=D_obs, D_in=D_latent + D_input))
std_model.add_data(dense_data, inputs=inputs)


# Run a Gibbs sampler
std_lls, std_x_smpls, std_smoothed_obss = \
    zip(*[gibbs_update(std_model) for _ in progprint_xrange(N_samples)])

# Plot the log likelihood over iterations
# plt.figure(figsize=(10,6))
# plt.plot(lls,'-b')
# plt.plot([0,N_samples], truemodel.log_likelihood() * np.ones(2), '-k')
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')

# Plot the smoothed observations
fig = plt.figure(figsize=(10,10))
N_subplots = min(D_obs, 6)

ylims = (-0.1, 1.1)
xlims = (0, min(T,1000))

n_to_plot = np.arange(min(N_subplots, D_obs))
for i,j in enumerate(n_to_plot):
    ax = fig.add_subplot(N_subplots,1,i+1)
    # Plot spike counts
    given_ts = np.where(dense_data[:,j]==1)[0]
    ax.plot(given_ts, np.ones_like(given_ts), 'ko', markersize=5)

    # Plot the inferred rate
    ax.plot([-10], [0], 'ko', lw=2, label="obs.")
    ax.plot(zi_smoothed_obss[-1][:, j], '-', color=colors[0], label="zero inflated")
    ax.plot(std_smoothed_obss[-1][:, j], '-', color=colors[1], label="standard")
    ax.plot(true_rate[:, j], '--k', lw=2, label="true rate")

    if i == 0:
        plt.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.5))

    if i == N_subplots - 1:
        plt.xlabel('time index')
    else:
        ax.set_xticklabels([])

    ax.set_xlim(xlims)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("$x_%d(t)$" % (j+1))

plt.savefig("aux/zeroinflation.png")
plt.show()

