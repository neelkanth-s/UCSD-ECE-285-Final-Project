import numpy as np

def generate_synthetic_data(M=100, N=1000, r=10, seed=42):

    rng = np.random.default_rng(seed)

    # Step 1: Generate a random r-dimensional subspace
    A = rng.standard_normal((M, r))

    # Obtain an orthonormal basis
    Q, _ = np.linalg.qr(A)

    # Step 2: Generate N random points in this subspace
    Z = rng.standard_normal((r, N))

    # Normalize each column to unit L2 norm
    Z = Z / np.linalg.norm(Z, axis=0, keepdims=True)

    # Project points into the M-dimensional ambient space
    X = Q @ Z

    return X, Q

def find_U(R, r):
    eigval, eigvec = np.linalg.eigh(R)
    U = eigvec[:,-r:]
    return U
def subspace_recovery_error(R, R_pertubed, r):
    U = find_U(R,r)
    U_perturbed = find_U(R_pertubed, r)
    I = np.eye(R.shape[0])
    numerator = np.linalg.norm((I-U_perturbed@U_perturbed.T)@U, 'fro')
    denominator = np.linalg.norm(U, 'fro')
    return numerator/denominator

def clip_R(X, lambda_vec):
    M, N = X.shape

    upper_indices = np.triu_indices(M)

    R_clipped = np.zeros((M, M))

    for i in range(N):

        # Outer product for individual i
        R_i = np.outer(X[:, i], X[:, i])

        # Clip upper triangular entries
        R_clipped[upper_indices] += np.clip(
            R_i[upper_indices],
            -lambda_vec,
            lambda_vec
        )

    # Restore symmetry
    R_clipped = (
        np.triu(R_clipped)
        + np.triu(R_clipped, 1).T
    )

    return R_clipped
def generate_lambda(M):
    K = M * (M + 1) // 2

    upper_indices = np.triu_indices(M)

    lambda_vec = np.where(
        upper_indices[0] == upper_indices[1],
        np.sqrt(1 / K),
        np.sqrt(1 / (2 * K))
    )

    return lambda_vec