from __future__ import division
import numpy as np


solve_psd = np.linalg.solve


def kf(init_mu,init_sigma,As,BBTs,Cs,DDTs,emissions):
    T, D_latent = emissions.shape[0], As[0].shape[0]

    filtered_mus = np.empty((T,D_latent))
    filtered_sigmas = np.empty((T,D_latent,D_latent))

    x = np.empty((T,D_latent))

    # messages forwards
    prediction_mu, prediction_sigma = init_mu, init_sigma
    for t, (A,BBT,C,DDT) in enumerate(zip(As,BBTs,Cs,DDTs)):
        # condition
        filtered_mus[t], filtered_sigmas[t] = \
            condition_on(prediction_mu,prediction_sigma,C,DDT,emissions[t])

        # predict
        prediction_mu, prediction_sigma = \
            A.dot(filtered_mus[t]), A.dot(filtered_sigmas[t]).dot(A.T) + BBT

    return filtered_mus, filtered_sigmas


def kf_resample_lds(init_mu,init_sigma,As,BBTs,Cs,DDTs,emissions):
    T, D_latent = emissions.shape[0], As[0].shape[0]
    x = np.empty((T,D_latent))

    filtered_mus, filtered_sigmas = kf(init_mu,init_sigma,As,BBTs,Cs,DDTs,emissions)

    # sample backwards
    # TODO pull rng out of the loop
    x[-1] = np.random.multivariate_normal(filtered_mus[-1],filtered_sigmas[-1])
    for t in xrange(T-2,-1,-1):
        x[t] = np.random.multivariate_normal(
            *condition_on(filtered_mus[t],filtered_sigmas[t],As[t],BBTs[t],x[t+1]))

    return x


def condition_on(mu_x,sigma_x,A,sigma_obs,y):
    # mu = mu_x + sigma_xy sigma_yy^{-1} (y - A mu_x)
    # sigma = sigma_x - sigma_xy sigma_yy^{-1} sigma_xy'
    sigma_xy = sigma_x.dot(A.T)
    sigma_yy = A.dot(sigma_x).dot(A.T) + sigma_obs
    mu = mu_x + sigma_xy.dot(solve_psd(sigma_yy, y - A.dot(mu_x)))
    sigma = sigma_x - sigma_xy.dot(solve_psd(sigma_yy,sigma_xy.T))
    return mu, symmetrize(sigma)


def symmetrize(A):
    ret = A+A.T
    ret /= 2.
    return ret

