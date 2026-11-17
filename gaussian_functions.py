import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def compute_mu0(epsilon, delta):
    """Largest mu with Q(eps/mu - mu/2) - e^eps Q(eps/mu + mu/2) <= delta  (problem P2 in [2])."""
    f = lambda mu: (norm.sf(epsilon / mu - mu / 2)
                    - np.exp(epsilon) * norm.sf(epsilon / mu + mu / 2) - delta)
    return brentq(f, 1e-6, 100.0, xtol=1e-12)


def gaussian_noise_iid(R, epsilon, delta, delta_2=1.0):
    """i.i.d. Gaussian (Lemma 1): same sigma = Delta_2 / mu0 on every upper-triangular entry.
    Delta_2 = 1 for unit-norm records (Dwork et al. 2014), applied to the unclipped R."""
    M = R.shape[0]
    upper_indices = np.triu_indices(M)
    sigma = delta_2 / compute_mu0(epsilon, delta)

    noise = np.zeros_like(R)
    noise[upper_indices] = np.random.normal(0, sigma, size=len(upper_indices[0]))
    noise = np.triu(noise) + np.triu(noise, 1).T
    return R + noise


def gaussian_noise_inid(R, epsilon, delta, lambda_vec):
    """i.n.i.d. Gaussian (Theorem 4): sigma_i^2 = lambda_i * ||lambda||_1 / mu0^2,
    applied to the clipped R."""
    M = R.shape[0]
    upper_indices = np.triu_indices(M)
    sigma = np.sqrt(lambda_vec * np.sum(lambda_vec)) / compute_mu0(epsilon, delta)

    noise = np.zeros_like(R)
    noise[upper_indices] = np.random.normal(0, sigma)
    noise = np.triu(noise) + np.triu(noise, 1).T
    return R + noise
