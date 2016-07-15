from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import AutoRegression, DiagonalRegression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import LDS

npr.seed(0)

from pybasicbayes.util.profiling import show_line_stats

#########################
#  set some parameters  #
#########################
D_obs, D_latent = 4, 4
mu_init = np.zeros(D_latent)
mu_init[0] = 1.0
sigma_init = 0.01*np.eye(D_latent)

def random_rotation(n,theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n,n))
    out[:2,:2] = rot
    q = np.linalg.qr(np.random.randn(n,n))[0]
    return q.dot(out).dot(q.T)

A = random_rotation(D_latent, np.pi/24.)
sigma_states = 0.01*np.eye(D_latent)

C = np.random.randn(D_obs, D_latent)
sigma_obs = 0.1 * np.eye(D_obs)


###################
#  generate data  #
###################

truemodel = LDS(
    dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
    emission_distn=DiagonalRegression(D_obs, D_latent, A=C, sigmasq=np.diag(sigma_obs)))

truemodel.mu_init = mu_init
truemodel.sigma_init = sigma_init

T = 1000
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

###############
#  make model #
###############
model = LDS(
    dynamics_distn=AutoRegression(
            nu_0=D_latent+3,
            S_0=D_latent*np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent)),
            K_0=D_latent*np.eye(D_latent)),
    emission_distn=DiagonalRegression(D_obs, D_latent, alpha_0=2.0, beta_0=1.0))
model.add_data(data=data, mask=mask)


###############
#  fit model  #
###############
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
# lls = [gibbs_update(model) for _ in progprint_xrange(N_samples)]

## EM -- initialized with a few Gibbs iterations
# [model.resample_model() for _ in progprint_xrange(100)]
# lls = [em_update(model) for _ in progprint_xrange(N_samples)]

## Mean field
# lls = [meanfield_update(model) for _ in progprint_xrange(N_samples)]

## SVI
delay = 10.0
forgetting_rate = 0.5
stepsizes = (np.arange(N_samples) + delay)**(-forgetting_rate)
minibatchsize = 500
# [model.resample_model() for _ in progprint_xrange(100)]
lls = [svi_update(model, stepsizes[itr], minibatchsize) for itr in progprint_xrange(N_samples)]


################
# likelihoods  #
################
plt.figure()
plt.plot(sigma_obs_smpls)
plt.xlabel("iteration")
plt.ylabel("sigma_obs")

plt.figure()
plt.plot(lls,'-b')
plt.plot([0,N_samples], truemodel.log_likelihood(data, mask) * np.ones(2), '-k')
plt.xlabel('iteration')
plt.ylabel('log likelihood')


################
#  smoothing   #
################
smoothed_obs = model.states_list[0].smooth()
sample_predictive_obs = model.states_list[0].stateseq.dot(model.C.T)

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
    # plt.plot(sample_predictive_obs, ':b', label="sample")

    plt.imshow(1-mask[:,i][None,:],cmap="Greys",alpha=0.25,extent=(0,T) + ylims, aspect="auto")

    if i == 0:
        plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.5))

    if i == N_subplots - 1:
        plt.xlabel('time index')

    plt.ylabel("$x_%d(t)$" % (i+1))
    plt.ylim(ylims)
    plt.xlim(xlims)
plt.savefig("missing_data_ex_lownoise.png")

################
#  predicting  #
################
# Nseed = 1700
# Npredict = 100
# prediction_seed = data[:Nseed]
#
# preds = model.sample_predictions(
#     prediction_seed, Npredict, obs_noise=False)
#
# plt.figure()
# plt.plot(given_data, 'k', label="observed")
# plt.plot(masked_data, ':k', label="masked")
# plt.plot(Nseed + np.arange(Npredict), preds, 'b')
# plt.xlabel('time index')
# plt.ylabel('prediction')

plt.show()

