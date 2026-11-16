import numpy as np
def laplace_noise_iid(R, epsilon, lambda_vec):
    M = R.shape[0]

    # Generate Laplace noise for upper triangular entries
    upper_indices = np.triu_indices(M)

    noise = np.zeros_like(R)
    delta_f = np.linalg.norm(lambda_vec, 1)
    noise[upper_indices] = np.random.laplace(
        loc=0,
        scale=delta_f / epsilon,
        size=len(upper_indices[0])
    )

    # Mirror upper triangle to lower triangle
    noise = np.triu(noise) + np.triu(noise, 1).T

    return R + noise
def laplace_noise_inid(R, epsilon, lambda_vec):
    M = R.shape[0]

    # Indices of upper triangular entries
    upper_indices = np.triu_indices(M)

    # Calculate optimal noise scales
    beta = (
        np.sum(lambda_vec**(2/3))
        * lambda_vec**(1/3)
        / epsilon
    )

    # Generate noise
    noise = np.zeros_like(R)

    noise[upper_indices] = np.random.laplace(
        loc=0,
        scale=beta
    )

    # Restore symmetry
    noise = np.triu(noise) + np.triu(noise, 1).T

    return R + noise

