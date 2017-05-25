from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression, DiagonalRegression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import DefaultLDS, MissingDataLDS

npr.seed(0)

# Model parameters
D_obs = 4
D_latent = 4
T = 1000

# Simulate from an LDS
truemodel = DefaultLDS(D_obs, D_latent)
data, stateseq = truemodel.generate(T)

# Mask off a chunk of data
mask = np.ones_like(data, dtype=bool)
chunksz = 100
for i,offset in enumerate(range(0,T,chunksz)):
    j = i % (D_obs + 1)
    if j < D_obs:
        mask[offset:min(offset+chunksz, T), j] = False
    if j == D_obs:
        mask[offset:min(offset+chunksz, T), :] = False

# Fit with another LDS
model = MissingDataLDS(
    dynamics_distn=Regression(
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent)),
    emission_distn=DiagonalRegression(D_obs, D_latent, alpha_0=2.0, beta_0=1.0))
model.add_data(data=data, mask=mask)


# Fit the model
N_samples = 500
sigma_obs_smpls = []
def gibbs_update(model):
    model.resample_model()
    sigma_obs_smpls.append(model.sigma_obs_flat)
    return model.log_likelihood()

def em_update(model):
    model.EM_step()
    sigma_obs_smpls.append(model.sigma_obs_flat)
    return model.log_likelihood()

def meanfield_update(model):
    model.meanfield_coordinate_descent_step()
    sigma_obs_smpls.append(model.emission_distn.mf_beta / model.emission_distn.mf_alpha)
    model.resample_from_mf()
    return model.log_likelihood()

def svi_update(model, stepsize, minibatchsize):
    # Sample a minibatch
    start = np.random.randint(0,T-minibatchsize+1)
    minibatch = data[start:start+minibatchsize]
    minibatch_mask = mask[start:start+minibatchsize]
    prob = minibatchsize/float(T)
    model.meanfield_sgdstep(minibatch, prob, stepsize, masks=minibatch_mask)

    sigma_obs_smpls.append(model.emission_distn.mf_beta / model.emission_distn.mf_alpha)
    model.resample_from_mf()
    return model.log_likelihood(data, mask=mask)


# Gibbs
lls = [gibbs_update(model) for _ in progprint_xrange(N_samples)]

## EM -- initialized with a few Gibbs iterations
# [model.resample_model() for _ in progprint_xrange(100)]
# lls = [em_update(model) for _ in progprint_xrange(N_samples)]

## Mean field
# lls = [meanfield_update(model) for _ in progprint_xrange(N_samples)]

## SVI
# delay = 10.0
# forgetting_rate = 0.5
# stepsizes = (np.arange(N_samples) + delay)**(-forgetting_rate)
# minibatchsize = 500
# # [model.resample_model() for _ in progprint_xrange(100)]
# lls = [svi_update(model, stepsizes[itr], minibatchsize) for itr in progprint_xrange(N_samples)]

# Plot the log likelihood
plt.figure()
plt.plot(lls,'-b')
dummymodel = MissingDataLDS(
    dynamics_distn=truemodel.dynamics_distn,
    emission_distn=truemodel.emission_distn)
plt.plot([0,N_samples], dummymodel.log_likelihood(data, mask=mask) * np.ones(2), '-k')
plt.xlabel('iteration')
plt.ylabel('log likelihood')

# Plot the inferred observation noise
plt.figure()
plt.plot(sigma_obs_smpls)
plt.xlabel("iteration")
plt.ylabel("sigma_obs")

# Smooth over missing data
smoothed_obs = model.states_list[0].smooth()
sample_predictive_obs = model.states_list[0].gaussian_states.dot(model.C.T)

plt.figure()
given_data = data.copy()
given_data[~mask] = np.nan
masked_data = data.copy()
masked_data[mask] = np.nan
ylims = (-1.1*abs(data).max(), 1.1*abs(data).max())
xlims = (0, min(T,1000))

N_subplots = min(D_obs,4)
for i in range(N_subplots):
    plt.subplot(N_subplots,1,i+1,aspect="auto")

    plt.plot(given_data[:,i], 'k', label="observed")
    plt.plot(masked_data[:,i], ':k', label="masked")
    plt.plot(smoothed_obs[:,i], 'b', lw=2, label="smoothed")

    plt.imshow(1-mask[:,i][None,:],cmap="Greys",alpha=0.25,extent=(0,T) + ylims, aspect="auto")

    if i == 0:
        plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.5))

    if i == N_subplots - 1:
        plt.xlabel('time index')

    plt.ylabel("$x_%d(t)$" % (i+1))
    plt.ylim(ylims)
    plt.xlim(xlims)

plt.show()

