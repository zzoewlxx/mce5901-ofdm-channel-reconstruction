"""Shared building blocks: pilot layout, conventional estimators, metrics.

Conventions (see README / proposal):
  * vectors are rows: H has shape (n_vectors, n_subcarriers), complex
  * pilot SNR is relative to the average channel power after ONE global scaling
    (never normalize each snapshot by its own true power)
  * NMSE is computed on the withheld (non-pilot) subcarriers only
"""
import numpy as np


def pilot_positions(n, n_obs):
    """n_obs approximately uniform pilot indices in a block of n, including both ends."""
    return np.unique(np.round(np.linspace(0, n - 1, n_obs)).astype(int))


def withheld(n, p):
    return np.setdiff1d(np.arange(n), p)


def add_noise(H_p, noise_var, seed):
    """Noisy pilot observations y = h_p + n, n ~ CN(0, noise_var)."""
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal(H_p.shape) + 1j * rng.standard_normal(H_p.shape)) / np.sqrt(2)
    return H_p + np.sqrt(noise_var) * z


def nmse_db(est, ref):
    return float(10 * np.log10(np.sum(np.abs(est - ref) ** 2) / np.sum(np.abs(ref) ** 2)))


# ------------------------------------------------------------- interpolation
def linear_matrix(n, p):
    """(n x len(p)) matrix W so that  H_est = Y @ W.T  is linear interpolation."""
    eye = np.eye(len(p))
    return np.stack([np.interp(np.arange(n), p, eye[j]) for j in range(len(p))], axis=1)


def spline_matrix(n, p):
    """Natural cubic-spline interpolation as an (n x len(p)) matrix (no scipy needed)."""
    x = np.asarray(p, float)
    m, h = len(x), np.diff(x)
    A = np.zeros((m, m))
    A[0, 0] = A[-1, -1] = 1.0
    for i in range(1, m - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
    xs = np.arange(n, dtype=float)
    seg = np.clip(np.searchsorted(x, xs, side="right") - 1, 0, m - 2)
    t0, t1, hi = xs - x[seg], x[seg + 1] - xs, h[seg]
    W = np.zeros((n, m))
    for j in range(m):
        y = np.zeros(m)
        y[j] = 1.0
        rhs = np.zeros(m)
        rhs[1:-1] = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        M = np.linalg.solve(A, rhs)
        W[:, j] = ((M[seg] * t1 ** 3 + M[seg + 1] * t0 ** 3) / (6 * hi)
                   + (y[seg] / hi - M[seg] * hi / 6) * t1 + (y[seg + 1] / hi - M[seg + 1] * hi / 6) * t0)
    return W


# --------------------------------------------------------------------- LMMSE
def channel_stats(H, chunk=200_000):
    """Mean and covariance C[j,k] = E[(h_j-mu_j)(h_k-mu_k)^*] of row vectors, in chunks."""
    n = H.shape[1]
    s1 = np.zeros(n, complex)
    s2 = np.zeros((n, n), complex)
    for i in range(0, len(H), chunk):
        X = H[i:i + chunk].astype(np.complex128)
        s1 += X.sum(0)
        s2 += X.T @ X.conj()
    mu = s1 / len(H)
    return mu, s2 / len(H) - np.outer(mu, mu.conj())


class LMMSE:
    """h_est = mu + C[:,p] (C[p,p] + (noise_var + rho * mean pilot power) I)^-1 (y - mu_p).

    With sample statistics this equals ridge regression from pilots to all subcarriers.
    """

    def __init__(self, mu, C, p):
        self.mu, self.C, self.p = mu, C, np.asarray(p)
        self.ppow = float(np.real(np.trace(C[np.ix_(p, p)])) / len(p))

    def matrix(self, noise_var, rho=1e-3):
        p = self.p
        Cpp = self.C[np.ix_(p, p)] + (noise_var + rho * self.ppow) * np.eye(len(p))
        return np.linalg.solve(Cpp.T, self.C[:, p].T).T

    def apply(self, Y, G):
        return self.mu + (Y - self.mu[self.p]) @ G.T
