"""Computational cost of ONE estimate for every estimator in the project: analytic multiply-accumulate
counts, stored parameters, and measured CPU time (batch throughput and single-vector latency).

Main line (one antenna, 64 subcarriers, 16 / 8 / 4 pilots; the withheld subcarriers are estimated):
  linear      linear interpolation as implemented: dense matrix  Y @ W.T  (experiments/01)
  spline      natural cubic spline as implemented: dense matrix  Y @ W.T
  lmmse       LMMSE with G precomputed for one assumed SNR (LMMSE.matrix / LMMSE.apply)
  lmmse_ps    LMMSE with each vector's own noise variance (LMMSE.apply_per_sample, eigen form)
  mlp         MLP 2x128 of experiments/02 (input 2|P|+1, output 2|M|), neural.make_predictor
  residual    the same MLP plus the per-sample LMMSE base (neural.make_predictor("residual"))
  gmm         GMM estimator of experiments/03 as implemented (gmm_feasibility.mix_estimate), K = 16
  gmm_pre     the same estimator with all per-component matrices precomputed (online part only)
Joint antenna x frequency setting (spacefreq of experiments/03: 32 antennas x 16 subcarriers = 512
entries; pilots on 32 or 8 antennas x 4 subcarriers = 128 or 32 pilots):
  joint_lmmse     LMMSE with the full 512 x 512 covariance, G precomputed
  joint_lmmse_ps  the same with each vector's own noise variance (apply_per_sample; base of jresidual)
  joint_mlp       MLP [2|P|+1, 512, 512, 2|M|] (placeholder size: --joint-hidden; jmlp of experiments/07)
  joint_residual  the same MLP plus the per-sample joint LMMSE base (jresidual of experiments/07)
  gmm, gmm_pre with K = 16 and 64

Counting rule (README.md has the derivations): a multiply-accumulate (MAC) is counted for every
multiplication in a matrix product, element-wise scaling or inner product; complex x complex = 1
complex MAC = 4 real MACs, complex x real = 2 real MACs, real x real = 1 real MAC; "complex MACs"
of real-valued work are real MACs / 4. A matrix product of complex data with a REAL matrix is
counted as complex x complex, because numpy promotes the matrix and calls zgemm (as implemented);
element-wise products and divisions are counted at their arithmetic cost (complex x real = 2), although
numpy's element-wise loops promote too (this understates the as-implemented element-wise work by at
most d/2 complex MACs per GMM component). Not counted: additions without a multiplication, the bias
add, ReLU, exp / log, the K divisions normalising the GMM posterior weights, the snr/30 input scaling
and the 1j recombination of the MLP output. Every formula is checked at start-up against counted
re-implementations whose outputs must equal the timed implementations (check_formulas).

Nothing here reads measured data: inputs, weights and covariances are synthetic with the right
shapes, because the run time of these dense operations does not depend on the values. No accuracy
is measured, so the validation blocks are not needed, and run 0057 is never involved.

    python experiments/05_complexity/run_complexity.py --member <name> --smoke         # seconds
    python experiments/05_complexity/run_complexity.py --member <name> --threads 1     # full run
"""
import os
import sys


def _threads_arg(argv, default="1"):
    """Value of --threads, read before numpy is imported (OpenBLAS reads the variable at load time)."""
    for i, x in enumerate(argv):
        if x == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--threads="):
            return x.split("=", 1)[1]
    return default


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = _threads_arg(sys.argv[1:])
import argparse  # noqa: E402
import gc  # noqa: E402
import platform  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import DEFAULT_NPZ  # noqa: E402
from estimators import LMMSE, linear_matrix, pilot_positions, spline_matrix, withheld  # noqa: E402
from mlp_numpy import MLP  # noqa: E402
from neural import VARIANTS, features, make_predictor  # noqa: E402
from runlog import save_run  # noqa: E402

MAIN_METHODS = ("linear", "spline", "lmmse", "lmmse_ps", "mlp", "residual", "gmm", "gmm_pre")
ALL_METHODS = MAIN_METHODS + ("joint_lmmse", "joint_lmmse_ps", "joint_mlp", "joint_residual")
ALIAS = {"joint_lmmse": "lmmse", "joint_lmmse_ps": "lmmse_ps", "joint_mlp": "mlp", "joint_residual": "residual"}
NAMES = {"linear": "Linear interpolation", "spline": "Cubic spline", "lmmse": "LMMSE, precomputed G",
         "lmmse_ps": "LMMSE, per-sample SNR", "mlp": "MLP {h}", "residual": "LMMSE + residual MLP {h}",
         "gmm": "GMM K={K} (mix_estimate)", "gmm_pre": "GMM K={K}, precomputed",
         "joint_lmmse": "Joint LMMSE, precomputed G", "joint_lmmse_ps": "Joint LMMSE, per-sample SNR",
         "joint_mlp": "Joint MLP {h}", "joint_residual": "Joint LMMSE + residual MLP {h}"}
FAMILY = {"linear": "linear", "spline": "spline", "lmmse": "lmmse", "lmmse_ps": "lmmse", "joint_lmmse": "lmmse",
          "joint_lmmse_ps": "lmmse", "mlp": "mlp", "joint_mlp": "mlp", "residual": "residual",
          "joint_residual": "residual", "gmm": "gmm", "gmm_pre": "gmm"}


def gmm_module():
    """experiments/03_gmm_feasibility/gmm_feasibility.py (imported lazily: it loads scikit-learn)."""
    sys.path.insert(0, os.path.join(ROOT, "experiments", "03_gmm_feasibility"))
    import gmm_feasibility
    return gmm_feasibility


# ------------------------------------------------------------------------------ analytic counts
def mlp_counts(sizes):
    """(real MACs, stored reals) of a dense network: weights a*b per layer, plus b biases."""
    pairs = list(zip(sizes[:-1], sizes[1:]))
    return sum(a * b for a, b in pairs), sum(a * b + b for a, b in pairs)


def spline_min_forms(P, M):
    """Real MACs and stored reals of the two cheap spline forms (real weights on complex data):
    dense rows for the withheld subcarriers only, and the local form (3-tap right-hand side,
    precomputed Thomas recursion for the |P|-2 interior second derivatives, 4 taps per output)."""
    dense = (2 * M * P, M * P)
    if P < 3:
        return dense, None
    ops = 3 * (P - 2) + (3 * (P - 2) - 2) + 4 * M          # complex x real multiplications
    return dense, (2 * ops, ops)


def counts(method, d, P, K=None, sizes=None):
    """Cost of ONE estimate of a d-entry vector from P pilots (M = d - P withheld entries).

    real / real_min: real MACs as implemented / minimal; params / params_min: stored real numbers
    needed online; setup (GMM as implemented only): approximate real MACs mix_estimate spends per
    CALL rebuilding its K inverses and gains (LU-based inverse ~4P^3/3, slogdet ~P^3/3 complex MACs).
    """
    M = d - P
    method = ALIAS.get(method, method)   # the joint methods run the same code as their main-line versions
    if method == "linear":      # numpy promotes the real W to complex; minimal: 2 real taps per output
        return dict(real=4 * d * P, real_min=4 * M, params=d * P, params_min=2 * M)
    if method == "spline":
        dense, local = spline_min_forms(P, M)
        best = dense if local is None or dense[0] <= local[0] else local
        return dict(real=4 * d * P, real_min=best[0], params=d * P, params_min=best[1])
    if method == "lmmse":        # mu + G (y - mu_P); minimal: withheld rows, mean folded in a bias
        return dict(real=4 * d * P, real_min=4 * M * P, params=2 * d * P + 2 * d, params_min=2 * M * P + 2 * M)
    if method == "lmmse_ps":     # z = U^H (y - mu_P) [P^2]; z / (lam + rho ppow + s2) [P x real]; mu + A z [dP]
        return dict(real=4 * P * P + 2 * P + 4 * d * P, real_min=4 * P * P + 2 * P + 4 * M * P,
                    params=2 * d + P + 2 * P * P + 2 * d * P + 1,          # mu, lam, U, A = C[:,P] U, rho*ppow
                    params_min=2 * P + 2 * M + P + 2 * P * P + 2 * M * P + 1)
    if method == "mlp":
        macs, params = mlp_counts(sizes)
        return dict(real=macs, real_min=macs, params=params, params_min=params)
    if method == "residual":
        a, b = counts("mlp", d, P, sizes=sizes), counts("lmmse_ps", d, P)
        return {k: a[k] + b[k] for k in a}
    if K is None or K < 2:
        raise ValueError(f"{method} needs K >= 2 (K = 1 is the plain LMMSE)")
    tri = P * (P + 1) // 2
    minimal = 4 * K * (tri + P + M * P)            # Cholesky-whitened likelihood + withheld rows only
    params_min = K * (P * P + 2 * M * P + 1)       # triangular L^-1 (real diagonal), B_k, log pi - log det
    if method == "gmm":          # per component: y^H S^-1 y [P^2 + P], G y [dP], weight x estimate [d x real]
        return dict(real=4 * K * (P * P + P + d * P) + 2 * K * d, real_min=minimal,
                    setup=4 * K * (d * P * P + 3 * P ** 3), params=K + 2 * K * d * d, params_min=params_min)
    if method == "gmm_pre":      # per component: w = L^-1 y [P^2], |w|^2 [P/2], weight x y [P/2], B (w y) [MP]
        return dict(real=4 * K * (P * P + P + M * P), real_min=minimal,
                    params=K * (2 * P * P + 2 * M * P + 1), params_min=params_min)
    raise ValueError(method)


# ----------------------------------------------------------------------- precomputed GMM (gmm_pre)
def gmm_precompute(P, M, pi, C, nv):
    """Everything mix_estimate rebuilds per call, for one assumed noise variance nv (its rho = 0):
    per component L_k^-1 with S_k = C_k[P,P] + nv I = L_k L_k^H, B_k = C_k[M,P] S_k^-1, and
    log pi_k - log det S_k (the -|P| log(pi) constant cancels in the posterior weights)."""
    K, p = len(pi), len(P)
    Linv = np.empty((K, p, p), complex)
    B = np.empty((K, len(M), p), complex)
    const = np.empty(K)
    for k in range(K):
        L = np.linalg.cholesky(C[k][np.ix_(P, P)] + nv * np.eye(p))
        Linv[k] = np.tril(np.linalg.inv(L))                  # exactly lower triangular
        B[k] = C[k][np.ix_(M, P)] @ (Linv[k].conj().T @ Linv[k])
        const[k] = np.log(pi[k]) - 2 * np.sum(np.log(np.real(np.diag(L))))
    return Linv, B, const


def gmm_pre_apply(Y, pre):
    """Same estimate as gmm_feasibility.mix_estimate on the withheld entries, online work only."""
    Linv, B, const = pre
    K = len(const)
    L = np.empty((len(Y), K))
    for k in range(K):
        w = Y @ Linv[k].T                                    # rows L_k^-1 y, so |w|^2 = y^H S_k^-1 y
        L[:, k] = const[k] - (w.real ** 2 + w.imag ** 2).sum(1)
    W = np.exp(L - L.max(1, keepdims=True))
    W /= W.sum(1, keepdims=True)
    est = np.zeros((len(Y), B.shape[1]), complex)
    for k in range(K):
        est += (Y * W[:, k, None]) @ B[k].T
    return est


# ------------------------------------------------------------------- counted implementations
class Count:
    """Products that also count real MACs (complex x complex 4, complex x real 2, real x real 1)."""

    def __init__(self):
        self.real = 0

    @staticmethod
    def _f(a, b):
        return 2 ** (int(np.iscomplexobj(a)) + int(np.iscomplexobj(b)))

    def mm(self, A, B, sparse=False):
        """A @ B; sparse=True counts only the nonzero entries of B (a sparse or triangular operator)."""
        self.real += A.shape[0] * (np.count_nonzero(B) if sparse else B.size) * self._f(A, B)
        return A @ B

    def mul(self, A, B):
        out = A * B
        self.real += out.size * self._f(A, B)
        return out

    def abs2(self, W):
        self.real += 2 * W.size
        return (W.real ** 2 + W.imag ** 2).sum(1)


def spline_local(Y, p, miss, c):
    """Natural cubic spline at the withheld subcarriers in O(|P| + |M|) per vector (counted)."""
    x = p.astype(float)
    m, h = len(x), np.diff(x)
    q = m - 2
    diag, sub, sup = 2 * (h[:-1] + h[1:]), h[:-1], h[1:]         # interior tridiagonal system
    inv, cp = np.empty(q), np.empty(q)
    inv[0] = 1 / diag[0]
    cp[0] = sup[0] * inv[0]
    for j in range(1, q):                                         # offline Thomas factors
        inv[j] = 1 / (diag[j] - sub[j] * cp[j - 1])
        cp[j] = sup[j] * inv[j]
    R = [c.mul(Y[:, i - 1], 6 / h[i - 1]) + c.mul(Y[:, i], -6 / h[i - 1] - 6 / h[i]) + c.mul(Y[:, i + 1], 6 / h[i])
         for i in range(1, m - 1)]
    dp = [c.mul(R[0], inv[0])]
    for j in range(1, q):
        dp.append(c.mul(R[j] - c.mul(dp[-1], sub[j]), inv[j]))
    sec = [None] * q
    sec[-1] = dp[-1]
    for j in range(q - 2, -1, -1):
        sec[j] = dp[j] - c.mul(sec[j + 1], cp[j])
    zero = np.zeros(len(Y), complex)
    S = np.column_stack([zero] + sec + [zero])                    # natural spline: zero at both ends
    xs = miss.astype(float)
    seg = np.searchsorted(x, xs, side="right") - 1
    t0, t1, hi = xs - x[seg], x[seg + 1] - xs, h[seg]
    return (c.mul(S[:, seg], t1 ** 3 / (6 * hi) - hi * t1 / 6) + c.mul(S[:, seg + 1], t0 ** 3 / (6 * hi) - hi * t0 / 6)
            + c.mul(Y[:, seg], t1 / hi) + c.mul(Y[:, seg + 1], t0 / hi))


def random_cov(rng, d, rank=32, out=None):
    """Random SPD covariance (low rank plus a small ridge), unit-order power."""
    r = min(d, rank)
    B = (rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))) / np.sqrt(2 * r)
    C = np.matmul(B, B.conj().T, out=out)
    i = np.arange(d)
    C[i, i] += 0.05
    return C


def complex_normal(rng, shape, chunk=20_000):
    out = np.empty(shape, complex)
    for i in range(0, shape[0], chunk):
        j = min(i + chunk, shape[0])
        out[i:j].real = rng.standard_normal((j - i,) + tuple(shape[1:]))
        out[i:j].imag = rng.standard_normal((j - i,) + tuple(shape[1:]))
    out *= np.sqrt(0.5)
    return out


def check_formulas(gf, hidden, N=3):
    """Counted re-implementations on small sizes: their MAC counts must equal the formulas, and their
    outputs must equal the implementations that are timed. Returns the names of the checked cases."""
    rng = np.random.default_rng(1)
    done = []

    def expect(name, c, formula, got, want):
        if c.real != N * formula:
            raise AssertionError(f"{name}: counted {c.real / N:g} real MACs per vector, formula gives {formula}")
        if not np.allclose(got, want, rtol=1e-8, atol=1e-10):
            raise AssertionError(f"{name}: counted implementation differs from the timed one")
        done.append(name)

    for n, n_obs in ((64, 16), (64, 8), (64, 4), (20, 5), (13, 3)):
        p = pilot_positions(n, n_obs)
        miss = withheld(n, p)
        P, M = len(p), len(miss)
        tag = f"n={n} P={P}"
        Y, s = complex_normal(rng, (N, P)), rng.uniform(0, 30, N)
        nv = 10 ** (-s / 10)
        W, Ws = linear_matrix(n, p), spline_matrix(n, p)
        cnt_l, cnt_s = counts("linear", n, P), counts("spline", n, P)
        c = Count()
        expect(f"linear {tag}", c, cnt_l["real"], c.mm(Y, W.T.astype(complex)), Y @ W.T)
        c = Count()
        expect(f"linear 2-tap {tag}", c, cnt_l["real_min"], c.mm(Y, W[miss].T, sparse=True), (Y @ W.T)[:, miss])
        c = Count()
        expect(f"spline {tag}", c, cnt_s["real"], c.mm(Y, Ws.T.astype(complex)), Y @ Ws.T)
        dense, local = spline_min_forms(P, M)
        c = Count()
        expect(f"spline dense-withheld {tag}", c, dense[0], c.mm(Y, Ws[miss].T), (Y @ Ws.T)[:, miss])
        c = Count()
        expect(f"spline local {tag}", c, local[0], spline_local(Y, p, miss, c), (Y @ Ws.T)[:, miss])
        if cnt_s["real_min"] != min(dense[0], local[0]):
            raise AssertionError("spline minimal count is not the cheaper form")

        mu = 0.1 * complex_normal(rng, (n,))
        lm = LMMSE(mu, random_cov(rng, n, rank=6), p)
        G = lm.matrix(0.1)
        cnt = counts("lmmse", n, P)
        c = Count()
        expect(f"lmmse {tag}", c, cnt["real"], mu + c.mm(Y - mu[p], G.T), lm.apply(Y, G))
        c = Count()
        expect(f"lmmse withheld {tag}", c, cnt["real_min"], (mu[miss] - G[miss] @ mu[p]) + c.mm(Y, G[miss].T),
               lm.apply(Y, G)[:, miss])
        want_ps = lm.apply_per_sample(Y, nv)
        lam, U, A = lm._eig

        def per_sample(c, rows):
            z = c.mm(Y - mu[p], U.conj())
            z = c.mul(z, 1 / (lam[None, :] + 1e-6 * lm.ppow + nv[:, None]))
            return mu[rows] + c.mm(z, A[rows].T)
        cnt = counts("lmmse_ps", n, P)
        c = Count()
        expect(f"lmmse_ps {tag}", c, cnt["real"], per_sample(c, np.arange(n)), want_ps)
        c = Count()
        expect(f"lmmse_ps withheld {tag}", c, cnt["real_min"], per_sample(c, miss), want_ps[:, miss])

        sizes = [2 * P + 1] + list(hidden) + [2 * M]
        for variant in ("mlp", "residual"):
            net = MLP(sizes, seed=0)
            c = Count()
            h = features(Y, s)
            for i, (Wl, b) in enumerate(zip(net.W, net.b)):
                h = c.mm(h, Wl) + b
                h = h if i == len(net.W) - 1 else np.maximum(h, 0)
            out = h[:, :M] + 1j * h[:, M:]
            if variant == "residual":
                out = out + per_sample(c, np.arange(n))[:, miss]   # as implemented: all n, then withheld
            expect(f"{variant} {tag}", c, counts(variant, n, P, sizes=sizes)["real"], out,
                   make_predictor(variant, net, lm, miss)(Y, s))
            if counts("mlp", n, P, sizes=sizes)["params"] != net.n_params():
                raise AssertionError("MLP parameter formula differs from MLP.n_params()")
    if counts("mlp", 64, 16, sizes=[33, 128, 128, 96])["real"] != 32_896:      # 33*128 + 128*128 + 128*96
        raise AssertionError("MLP 2x128 with 16 pilots must have 32,896 real MACs")
    done.append("mlp 2x128, 16 pilots = 32,896 real MACs")

    if gf is not None:
        for d, n_obs, K in ((24, 6, 3), (64, 8, 4), (40, 10, 5)):
            P = np.sort(rng.choice(d, n_obs, replace=False))
            M = np.setdiff1d(np.arange(d), P)
            tag = f"d={d} P={n_obs} K={K}"
            C = np.stack([random_cov(rng, d, rank=4) for _ in range(K)])
            pi, nv = rng.dirichlet(np.ones(K)), 0.1
            Y = complex_normal(rng, (N, n_obs))
            ref = gf.mix_estimate(Y, P, pi, C, nv)
            S = [C[k][np.ix_(P, P)] + nv * np.eye(n_obs) for k in range(K)]
            Si = [np.linalg.inv(x) for x in S]
            c = Count()                                  # per-vector work of mix_estimate (setup excluded)
            L = np.stack([np.log(pi[k]) - np.linalg.slogdet(S[k])[1] - c.mul(c.mm(Y.conj(), Si[k]), Y).sum(1).real
                          for k in range(K)], 1)
            Wt = np.exp(L - L.max(1, keepdims=True))
            Wt /= Wt.sum(1, keepdims=True)
            est = sum(c.mul(Wt[:, k, None], c.mm(Y, (C[k][:, P] @ Si[k]).T)) for k in range(K))
            expect(f"gmm {tag}", c, counts("gmm", d, n_obs, K=K)["real"], est, ref)
            pre = gmm_precompute(P, M, pi, C, nv)
            if not np.allclose(gmm_pre_apply(Y, pre), ref[:, M], rtol=1e-8, atol=1e-10):
                raise AssertionError("gmm_pre_apply differs from mix_estimate")
            for name, tri, key in (("gmm_pre", False, "real"), ("gmm minimal (triangular)", True, "real_min")):
                Linv, B, const = pre
                c = Count()
                L = np.stack([const[k] - c.abs2(c.mm(Y, Linv[k].T, sparse=tri)) for k in range(K)], 1)
                Wt = np.exp(L - L.max(1, keepdims=True))
                Wt /= Wt.sum(1, keepdims=True)
                est = sum(c.mm(c.mul(Y, Wt[:, k, None]), B[k].T) for k in range(K))
                expect(f"{name} {tag}", c, counts("gmm_pre", d, n_obs, K=K)[key], est, ref[:, M])
    return done


# --------------------------------------------------------------------------------------- timing
def time_batch(fn, args, n, chunk, repeats, min_repeats, budget_s):
    """Per-vector wall time (us) when all n vectors are streamed through fn in chunks; one warm-up chunk.
    Also the process CPU time over all repeats: with one thread, cpu / wall well below 1 means the
    process was pre-empted (a loaded machine) and the wall times are too large."""
    fn(*(x[:chunk] for x in args))
    times, t_start, c_start = [], time.perf_counter(), time.process_time()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t = time.perf_counter()
            for i in range(0, n, chunk):
                fn(*(x[i:i + chunk] for x in args))
            times.append(time.perf_counter() - t)
            if len(times) >= min_repeats and time.perf_counter() - t_start > budget_s:
                break
    finally:
        if enabled:
            gc.enable()
    cpu_s = time.process_time() - c_start
    us = [1e6 * t / n for t in times]
    return {"median": float(np.median(us)), "min": float(np.min(us)), "all": us,
            "wall_s": float(sum(times)), "cpu_s": float(cpu_s)}


def time_latency(fn, args, calls, budget_s, warmup, min_calls):
    """Wall time (us) of single-vector calls (batch size 1), cycling over 256 different vectors."""
    rows = [tuple(x[i:i + 1] for x in args) for i in range(min(256, len(args[0])))]
    for i in range(warmup):
        fn(*rows[i % len(rows)])
    ts, t_start = [], time.perf_counter()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(calls):
            t = time.perf_counter_ns()
            fn(*rows[i % len(rows)])
            ts.append(time.perf_counter_ns() - t)
            if len(ts) >= min_calls and time.perf_counter() - t_start > budget_s:
                break
    finally:
        if enabled:
            gc.enable()
    us = np.array(ts) / 1e3
    return {"median": float(np.median(us)), "p90": float(np.percentile(us, 90)), "calls": len(ts)}


def blas_info():
    info = {"env_OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}
    try:
        cfg = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        info.update(name=cfg.get("name"), version=cfg.get("version"), config=cfg.get("openblas configuration"))
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_info
        info["threads_reported"] = [x.get("num_threads") for x in threadpool_info() if x.get("user_api") == "blas"]
    except Exception:
        pass
    return info


def cpu_name():
    """CPU model string (platform.processor() on Windows only gives the family / model numbers)."""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def spacefreq_pilots(n_pilots):
    """Pilot entries of experiments/03 spacefreq (entry = antenna*16 + subcarrier): an antenna subset
    with equal index stride x the 4 pilot subcarriers pilot_positions(16, 4)."""
    n_ant = n_pilots // 4
    if n_pilots % 4 or n_ant < 1 or 32 % n_ant:
        raise ValueError(f"spacefreq pilots must be 4 x a divisor of 32, got {n_pilots}")
    ants = np.arange(0, 32, 32 // n_ant)
    return (ants[:, None] * 16 + pilot_positions(16, 4)[None, :]).ravel()


def random_gmm(rng, K, d, shared):
    """K random SPD component covariances; shared=True (smoke only) gives K views of ONE matrix to save memory."""
    pi = rng.dirichlet(np.ones(K))
    if shared:
        return pi, np.broadcast_to(random_cov(rng, d), (K, d, d))
    C = np.empty((K, d, d), complex)
    for k in range(K):
        random_cov(rng, d, out=C[k])
    return pi, C


# ----------------------------------------------------------------------------------------- output
def fmt(x):
    if x is None:
        return "-"
    x = float(x)
    if x.is_integer():
        return f"{int(x):,}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def fmt_us(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.3g}")


def label(m):
    h = m.get("hidden")
    hs = None if h is None else (f"{len(h)}x{h[0]}" if len(set(h)) == 1 else "-".join(map(str, h)))
    return NAMES[m["method"]].format(K=m.get("K"), h=hs)


def markdown_table(metrics, threads):
    lines = [f"Per estimate (one vector); CPU time with {threads} BLAS thread(s); '-' = minimal equals as implemented.",
             "",
             "| Setting | Pilots | Method | Complex MACs | min. | Real MACs | Params (real) | min. "
             "| Batch us/vector | Latency us |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    notes = []
    for m in metrics.values():
        same = lambda a, b: None if a == b else b
        lines.append(f"| {m['setting']} | {m['pilots']} | {label(m)} | {fmt(m['macs_complex'])} "
                     f"| {fmt(same(m['macs_complex'], m['macs_complex_min']))} | {fmt(m['macs_real'])} "
                     f"| {fmt(m['params'])} | {fmt(same(m['params'], m['params_min']))} "
                     f"| {fmt_us(m['us_per_vector_batch'])} | {fmt_us(m['us_latency'])} |")
        if "setup_macs_complex_per_call" in m:
            notes.append(f"{m['setting']}/{m['pilots']} {label(m)}: mix_estimate also spends about "
                         f"{fmt(m['setup_macs_complex_per_call'])} complex MACs per CALL rebuilding its inverses "
                         "(included in the latency; spread over the chunk in the batch figure).")
    return "\n".join(lines + ([""] + [f"- {x}" for x in notes] if notes else []))


def plot(metrics, path, threads):
    from style import COL_W, COLORS, LABELS, MARKERS, apply_style
    plt = apply_style("paper")
    fig, ax = plt.subplots(figsize=(COL_W, 3.3))
    seen = set()
    for m in metrics.values():
        f = FAMILY[m["method"]]
        rebuild = m["method"] == "gmm"         # mix_estimate: its time also contains the per-call rebuild
        key = "gmm_rebuild" if rebuild else f
        name = "GMM, rebuilt per call" if rebuild else ("GMM, precomputed" if f == "gmm" else LABELS[f])
        filled = m["setting"] == "freq64"
        ax.scatter(m["macs_real"], m["us_per_vector_batch"], s=20, marker="h" if rebuild else MARKERS[f],
                   linewidths=0.9, edgecolors=COLORS[f], facecolors=COLORS[f] if filled else "none",
                   label=name if key not in seen else None, zorder=3)
        seen.add(key)
    x = np.array([m["macs_real"] for m in metrics.values()], float)
    y = np.array([m["us_per_vector_batch"] for m in metrics.values()], float)
    big = np.array([m["macs_real"] >= 1e5 and m["method"] != "gmm" for m in metrics.values()])
    if big.any():           # MAC rate from the large dense methods whose time is the counted work
        ns_per_mac = float(np.median(1e3 * y[big] / x[big]))
        xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 2)
        ax.plot(xx, xx * ns_per_mac / 1e3, ls="--", lw=0.8, color=COLORS["genie"], zorder=2,
                label=f"{1 / ns_per_mac:.1f} G real MAC/s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Real MACs per estimate (as implemented)")
    ax.set_ylabel("Time per vector, batch (µs)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.0)       # below the axes: no point is hidden
    ax.text(0.98, 0.03, f"filled: 64 subcarriers; open: 512-entry joint\n{threads} BLAS thread(s), synthetic inputs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#52514e")
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def main():
    # allow_abbrev=False: an abbreviated --thr would reach argparse but not _threads_arg (read before numpy)
    ap = argparse.ArgumentParser(description="Analytic MAC counts and measured CPU time per estimate.",
                                 allow_abbrev=False)
    ap.add_argument("--member", required=True)
    ap.add_argument("--data", default=DEFAULT_NPZ,
                    help="not read: every input is synthetic (accepted so all experiment scripts share one interface)")
    ap.add_argument("--threads", type=int, default=1, help="OpenBLAS threads (set before numpy is imported)")
    ap.add_argument("--smoke", action="store_true", help="tiny vector / repeat counts; finishes in seconds; not saved")
    ap.add_argument("--settings", nargs="*", default=["freq64", "spacefreq"], choices=["freq64", "spacefreq"])
    ap.add_argument("--methods", nargs="*", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    ap.add_argument("--densities", nargs="*", type=int, default=[16, 8, 4], help="pilots of 64 subcarriers")
    ap.add_argument("--spacefreq-pilots", nargs="*", type=int, default=[128, 32], help="32 ant x 4 sc, 8 ant x 4 sc")
    ap.add_argument("--gmm-k-freq64", nargs="*", type=int, default=[16])
    ap.add_argument("--gmm-k-spacefreq", nargs="*", type=int, default=[16, 64])
    ap.add_argument("--mlp-hidden", nargs="*", type=int, default=list(VARIANTS["mlp"][0]))
    ap.add_argument("--joint-hidden", nargs="*", type=int, default=[512, 512], help="placeholder joint MLP size")
    ap.add_argument("--snr-db", type=float, default=10.0, help="assumed SNR of the fixed-SNR estimators")
    ap.add_argument("--n-batch", type=int, default=100_000, help="vectors per batch-throughput repeat")
    ap.add_argument("--chunk", type=int, default=10_000, help="vectors per call in the batch measurement")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min-repeats", type=int, default=3, help="repeats kept even when the budget is exceeded")
    ap.add_argument("--batch-budget-s", type=float, default=300.0, help="stop repeating after this, per method")
    ap.add_argument("--latency-calls", type=int, default=1000)
    ap.add_argument("--latency-budget-s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads < 1:
        ap.error("--threads must be >= 1 (OPENBLAS_NUM_THREADS=0 would mean 'all cores')")
    if os.environ.get("OPENBLAS_NUM_THREADS") != str(a.threads):
        raise RuntimeError("OPENBLAS_NUM_THREADS was not set from --threads before numpy was imported")
    if min(a.gmm_k_freq64 + a.gmm_k_spacefreq, default=2) < 2:
        raise ValueError("GMM needs K >= 2 (K = 1 is the plain LMMSE, timed as lmmse / joint_lmmse)")
    if a.smoke:
        a.n_batch, a.chunk, a.repeats, a.min_repeats, a.latency_calls = 200, 100, 1, 1, 3
    a.chunk = min(a.chunk, a.n_batch)
    lat_warm, lat_min = (1, 1) if a.smoke else (20, 20)
    log = lambda s: print(s, flush=True)
    t_start = time.time()
    blas = blas_info()
    log(f"threads {a.threads}, BLAS {blas.get('name')} {blas.get('version')}, reported {blas.get('threads_reported')}")
    if blas.get("threads_reported") and set(blas["threads_reported"]) != {a.threads}:
        msg = f"BLAS reports {blas['threads_reported']} threads, requested {a.threads}"
        if not a.smoke:     # the timings would be labelled with the wrong thread count
            raise RuntimeError(msg + " (was numpy imported before OPENBLAS_NUM_THREADS was set?)")
        log("warning: " + msg)

    want_gmm = any(m in a.methods for m in ("gmm", "gmm_pre"))
    gf = gmm_module() if want_gmm else None
    checked = check_formulas(gf, a.mlp_hidden)
    log(f"MAC formulas match the counted implementations ({len(checked)} cases)")

    rng = np.random.default_rng(a.seed)
    nv0 = 10 ** (-a.snr_db / 10)
    N = a.n_batch
    metrics, warnings = {}, []

    def measure(setting, method, P, d, fn, args, K=None, hidden=None, sizes=None):
        cnt = counts(method, d, P, K=K, sizes=sizes)
        b = time_batch(fn, args, N, a.chunk, a.repeats, a.min_repeats, a.batch_budget_s)
        lat = time_latency(fn, args, a.latency_calls, a.latency_budget_s, lat_warm, lat_min)
        name = method if K is None else method.replace("gmm", f"gmm_k{K}")
        r4 = lambda v: float(f"{v:.4g}")
        m = {"setting": setting, "method": method, "pilots": int(P), "d": int(d), "withheld": int(d - P),
             "macs_complex": cnt["real"] / 4, "macs_real": cnt["real"],
             "macs_complex_min": cnt["real_min"] / 4, "macs_real_min": cnt["real_min"],
             "params": cnt["params"], "params_min": cnt["params_min"],
             "us_per_vector_batch": r4(b["median"]), "us_per_vector_batch_min": r4(b["min"]),
             "batch_repeats_us": [r4(v) for v in b["all"]],
             "us_latency": r4(lat["median"]), "us_latency_p90": r4(lat["p90"]), "latency_calls": lat["calls"],
             "batch_wall_s": r4(b["wall_s"]), "batch_cpu_s": r4(b["cpu_s"])}
        cpu_ratio = b["cpu_s"] / b["wall_s"] if b["wall_s"] > 0 else float("nan")
        if a.threads == 1 and b["wall_s"] >= 2 and cpu_ratio < 0.85:
            warnings.append(f"{setting}/{P}/{method}{'' if K is None else f' K={K}'}: process CPU time is only "
                            f"{cpu_ratio:.0%} of the wall time (pre-empted? the batch time is too large)")
        if K is not None:
            m["K"] = K
        if "setup" in cnt:
            m["setup_macs_complex_per_call"] = cnt["setup"] / 4
        if hidden is not None:
            m["hidden"], m["layer_sizes"] = list(hidden), list(sizes)
        metrics[f"{setting}/{P}/{name}"] = m
        log(f"  {setting:9s} {P:3d} pilots {name:14s} real MACs {cnt['real']:>12,}  batch {b['median']:10.3f} us/vector "
            f"({len(b['all'])} rep, cpu/wall {f'{cpu_ratio:.2f}' if b['wall_s'] >= 1 else 'n/a'})  "
            f"latency {lat['median']:10.1f} us ({lat['calls']} calls)")

    def gmm_methods(setting, p, miss, d, Ks, Y):
        for K in Ks:
            if not want_gmm:
                break
            pi, C = random_gmm(rng, K, d, shared=a.smoke and K * d * d * 16 > 32e6)
            if "gmm" in a.methods:
                measure(setting, "gmm", len(p), d, lambda X: gf.mix_estimate(X, p, pi, C, nv0), (Y,), K=K)
            pre = gmm_precompute(p, miss, pi, C, nv0) if "gmm_pre" in a.methods else None
            if pre is not None:     # the timed precomputed form must give mix_estimate's estimate
                ref = gf.mix_estimate(Y[:8], p, pi, C, nv0)[:, miss]
                err = float(np.max(np.abs(gmm_pre_apply(Y[:8], pre) - ref)) / np.max(np.abs(ref)))
                if not err < 1e-9:
                    raise AssertionError(f"gmm_pre differs from mix_estimate ({setting}, P={len(p)}, K={K}): {err:.1e}")
                checked.append(f"gmm_pre = mix_estimate on the timed {setting} P={len(p)} K={K} model "
                               f"(max rel. error {err:.1e})")
            del C
            if pre is not None:
                measure(setting, "gmm_pre", len(p), d, lambda X: gmm_pre_apply(X, pre), (Y,), K=K)
            del pre

    if "freq64" in a.settings:
        n = 64
        for n_obs in a.densities:
            p = pilot_positions(n, n_obs)
            miss = withheld(n, p)
            P, M = len(p), len(miss)
            Y, s = complex_normal(rng, (N, P)), rng.uniform(0, 30, N)
            nv = 10 ** (-s / 10)
            lm = LMMSE(0.01 * complex_normal(rng, (n,)), random_cov(rng, n), p)
            if "linear" in a.methods:
                W = linear_matrix(n, p)
                measure("freq64", "linear", P, n, lambda X: X @ W.T, (Y,))
            if "spline" in a.methods:
                Ws = spline_matrix(n, p)
                measure("freq64", "spline", P, n, lambda X: X @ Ws.T, (Y,))
            if "lmmse" in a.methods:
                G = lm.matrix(nv0)
                measure("freq64", "lmmse", P, n, lambda X: lm.apply(X, G), (Y,))
            if "lmmse_ps" in a.methods:
                measure("freq64", "lmmse_ps", P, n, lm.apply_per_sample, (Y, nv))
            for v in ("mlp", "residual"):
                if v in a.methods:
                    net = MLP([2 * P + 1] + list(a.mlp_hidden) + [2 * M], seed=a.seed)
                    measure("freq64", v, P, n, make_predictor(v, net, lm, miss), (Y, s),
                            hidden=a.mlp_hidden, sizes=net.sizes)
            gmm_methods("freq64", p, miss, n, a.gmm_k_freq64, Y)
            del Y, s, nv

    if "spacefreq" in a.settings:
        d = 512
        for n_pil in a.spacefreq_pilots:
            p = spacefreq_pilots(n_pil)
            miss = np.setdiff1d(np.arange(d), p)
            P, M = len(p), len(miss)
            Y, s = complex_normal(rng, (N, P)), rng.uniform(0, 30, N)
            nv = 10 ** (-s / 10)
            lmj = (LMMSE(np.zeros(d, complex), random_cov(rng, d), p)
                   if any(m in a.methods for m in ("joint_lmmse", "joint_lmmse_ps", "joint_residual")) else None)
            if "joint_lmmse" in a.methods:
                G = lmj.matrix(nv0)
                measure("spacefreq", "joint_lmmse", P, d, lambda X: lmj.apply(X, G), (Y,))
                del G
            if "joint_lmmse_ps" in a.methods:
                measure("spacefreq", "joint_lmmse_ps", P, d, lmj.apply_per_sample, (Y, nv))
            for method, variant in (("joint_mlp", "mlp"), ("joint_residual", "residual")):
                if method in a.methods:     # experiments/07 jmlp / jresidual: same features and predictor
                    net = MLP([2 * P + 1] + list(a.joint_hidden) + [2 * M], seed=a.seed)
                    measure("spacefreq", method, P, d, make_predictor(variant, net, lmj, miss), (Y, s),
                            hidden=a.joint_hidden, sizes=net.sizes)
                    del net
            del lmj
            gmm_methods("spacefreq", p, miss, d, a.gmm_k_spacefreq, Y)
            del Y, s, nv

    table = markdown_table(metrics, a.threads)
    print("\n" + table + "\n", flush=True)
    for w in warnings:
        log("warning: " + w)
    out_dir = tempfile.mkdtemp(prefix="complexity_")
    table_path = os.path.join(out_dir, "complexity_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    files = [table_path]
    fig_path = os.path.join(out_dir, "macs_vs_time.png")
    try:  # a plotting problem must never lose the measurements
        plot(metrics, fig_path, a.threads)
        files.append(fig_path)
    except Exception as e:
        log(f"plot failed ({e}); metrics are still saved")
    config = {"threads": a.threads, "blas": blas, "processor": platform.processor(), "cpu_name": cpu_name(),
              "machine": platform.machine(),
              "cpu_count": os.cpu_count(), "numpy": np.__version__, "python": platform.python_version(),
              "settings": a.settings, "methods": a.methods,
              "freq64": {"n": 64, "densities": a.densities, "mlp_hidden": a.mlp_hidden, "gmm_k": a.gmm_k_freq64},
              "spacefreq": {"d": 512, "pilots": a.spacefreq_pilots, "pattern": "antenna stride 32/(pilots/4) x "
                            "pilot_positions(16, 4); entry = antenna*16 + subcarrier",
                            "gmm_k": a.gmm_k_spacefreq, "joint_hidden": a.joint_hidden},
              "assumed_snr_db": a.snr_db, "n_batch": N, "chunk": a.chunk, "repeats": a.repeats,
              "min_repeats": a.min_repeats, "batch_budget_s": a.batch_budget_s, "latency_calls": a.latency_calls,
              "latency_budget_s": a.latency_budget_s, "latency_warmup": lat_warm, "seed": a.seed,
              "inputs": "synthetic: unit complex Gaussian pilots, random SPD covariances (rank 32 + 0.05 I), "
                        "He-initialised MLP weights; no measured data is read",
              "counting": "complex x complex = 1 complex MAC = 4 real MACs; complex x real = 2 real; "
                          "real x real = 1 real; macs_complex = macs_real / 4; a real matrix times complex data "
                          "counts as complex x complex (numpy promotes it to zgemm); element-wise products at "
                          "arithmetic cost; not counted: adds, bias, ReLU, exp/log, GMM weight normalisation, "
                          "snr/30 input scaling, 1j output recombination",
              "formula_check": {"passed": True, "cases": checked}, "timing_warnings": warnings, "smoke": a.smoke,
              "runtime_s": round(time.time() - t_start, 1)}
    if a.smoke:
        log(f"smoke test finished (nothing saved to results/); table {table_path}; "
            f"figure {fig_path if fig_path in files else '(plot failed)'}")
        return
    save_run("complexity", a.member, config, metrics, files=files,
             note="analytic MAC counts and measured CPU time per estimate on synthetic inputs; no data, run 0057 untouched")


if __name__ == "__main__":
    main()
