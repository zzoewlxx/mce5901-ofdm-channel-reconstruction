"""GMM estimator versus LMMSE with three seeds, a grid that brackets experiment 03's choice, and EM run up to
50 iterations: the feasibility numbers of experiments/03_gmm_feasibility made presentable.

Experiment 03 found GMM gains of +1.5 to +6 dB over LMMSE once antennas are included, but from one k-means
seed, with the chosen (K, alpha) at the corner of its grid in every row, EM stopped by a 20-iteration cap
while the log-likelihood was still rising (space32), and results written outside save_run. This script
repeats the comparison with 03's data loader, split, scale, noise realisations and selection protocol, and
  * fits every (K, alpha) of the grid for each seed in --seeds; the seed sets the k-means initialisation
    and, when --em-max is below the number of training vectors, the EM subsample
  * EM up to 50 iterations with 03's stopping rule (relative change of the mean log-likelihood < 1e-5);
    per fit: iterations, log-likelihood trajectory, cap hit, posterior hardness on the training vectors
    and on 500 clean even-block validation vectors (diagnostic only)
  * per seed and (pattern, SNR): LMMSE = K=1 with the global covariance of ALL training vectors and its
    loading chosen on even blocks (exactly as 03); GMM config chosen on even blocks; odd-block NMSE; gain
    over LMMSE with a paired block-bootstrap 95% CI (B=1000, the same resampled blocks for both methods and
    every seed); the chosen config and whether it is at the edge of the grid
  * over seeds: mean, min and max gain, the envelope of the per-seed CIs
Run 0057 is never loaded (03's loader reads runs 0053-0056 only). GMM covariances are not saved (spacefreq
K=64 is 268 MB per fit): a fit is deterministic given data, seed, code and BLAS thread count, and run.json
records a SHA-256 of every fitted model and k-means labelling so a refit can be checked against it.

Local EM (fit_em) = 03's fit_gmm (same k-means start, M-step with shrinkage, E-step, stopping rule) with the
diagnostics added and speed-ups that leave the model unchanged; check_against_03() compares it with
fit_gmm at start-up and stops the run if they disagree:
  * d <= 64 (space32, freq64): every quadratic form x^H C_k^-1 x and every weighted covariance of a chunk
    comes from ONE real matrix product with the upper triangle of x x^H, instead of K memory-bound passes
  * d > 64 (spacefreq): a component's covariance skips training vectors with responsibility < 1e-10
    (posteriors are practically 0/1 there; the largest skipped weight fraction is recorded)
  * both: responsibilities below 1e-200 are set to exactly 0. Otherwise they underflow to subnormal numbers,
    which make the matrix products that use them about 80x slower (03's code has this slow-down)

    python experiments/06_gmm_seeds/run_gmm_seeds.py --member <name>                  # spacefreq + space32
    python experiments/06_gmm_seeds/run_gmm_seeds.py --member <name> --smoke          # code test, not saved
    python experiments/06_gmm_seeds/run_gmm_seeds.py --member <name> --synthetic 2400 # synthetic file, not saved
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # small inv/slogdet are 50-180x slower with the default
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")      # silences a harmless sklearn warning on Windows
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True                         # importing experiment 03 must not write into its folder
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "03_gmm_feasibility"))
from dataset import DEFAULT_NPZ  # noqa: E402
from estimators import add_noise  # noqa: E402
from gmm_feasibility import RHOS, SNRS, cov, fit_gmm, load, logpdf, mix_estimate  # noqa: E402
from runlog import save_run  # noqa: E402

GRIDS = {   # (K, alpha). 03 chose K=16/alpha=0.05 in 8/8 spacefreq rows and K=64/alpha=0.05 in 12/12 space32 rows
    "spacefreq": [(8, 0.05), (16, 0.01), (16, 0.05), (32, 0.01), (32, 0.05), (64, 0.05)],
    "space32": [(16, 0.05), (64, 0.05), (128, 0.01), (128, 0.05)],
    "freq64": [(4, 0.05), (16, 0.05), (64, 0.05)],
}
SMOKE_GRID = [(4, 0.05), (8, 0.05)]
ITERS, TOL = 50, 1e-5          # EM cap and 03's stopping rule |ll_t - ll_t-1| < TOL |ll_t-1|
B = 1000                       # bootstrap resamples of the report blocks
FEATURE_MAX_D = 64             # d <= 64: one matrix product per chunk for all components (see module doc)
DROP = 1e-10                   # d > 64: responsibilities below this are left out of a component's covariance
KMEANS_THREADS = 4             # fixed so the k-means labels do not depend on the machine's core count
HELDOUT = 500                  # clean even-block validation vectors for the posterior-hardness diagnostic
EM_MAX_03 = 200_000            # 03's EM subsample size (its random draw precedes the bootstrap draw)
FLUSH = 1e-200                 # responsibilities below this are set to 0: subnormal numbers (< 2.2e-308)
                               # make the following matrix products ~80x slower; the effect is below rounding


# ----------------------------------------------------------------------------- EM
def kmeans_labels(X, Cg, K, seed, chunk=4096):
    """03's initialisation: k-means on unit-norm vectors with the common phase removed (same arithmetic)."""
    d = X.shape[1]
    v = np.linalg.eigh(Cg)[1][:, -1]
    ph = np.exp(-1j * np.angle(X @ v.conj()))
    nrm = np.linalg.norm(X, axis=1)
    Z = np.empty((len(X), 2 * d), np.float32)          # [Re F, Im F], built in chunks to save memory
    for i in range(0, len(X), chunk):
        F = X[i:i + chunk] * ph[i:i + chunk, None] / nrm[i:i + chunk, None]
        Z[i:i + chunk, :d], Z[i:i + chunk, d:] = F.real, F.imag
    with threadpool_limits(KMEANS_THREADS, user_api="openmp"):
        return KMeans(K, n_init=1, random_state=seed).fit_predict(Z)


def _posterior(L):
    """Per-row log-likelihood (log-sum-exp) and responsibilities exp(L - ll), with tiny entries flushed to 0."""
    m = L.max(1, keepdims=True)
    E = np.exp(L - m)
    E[E < FLUSH] = 0.0
    s = E.sum(1)
    return m[:, 0] + np.log(s), E / s[:, None]


def _hardness(L, ll):
    """Posterior hardness from log-joint L (n, K) and log-likelihood ll (n,): argmax, max posterior, and the
    margin (nats) between the best and the second-best component."""
    top2 = np.partition(L, L.shape[1] - 2, axis=1)[:, -2:] if L.shape[1] > 1 else np.zeros((len(L), 2))
    return {"arg": L.argmax(1), "maxpost": np.exp(L.max(1) - ll), "margin": top2[:, 1] - top2[:, 0]}


def _summary(h, labels=None):
    s = {"mean_max_posterior": round(float(h["maxpost"].mean()), 4),
         "frac_max_posterior_above_0.99": round(float(np.mean(h["maxpost"] > 0.99)), 4),
         "median_margin_nats": round(float(np.median(h["margin"])), 2)}
    if labels is not None:
        s["frac_changed_from_kmeans"] = round(float(np.mean(h["arg"] != labels)), 4)
    return s


def _em_direct(X, K, alpha, labels, Cg, eye, iters, tol, chunk=4096):
    """d > 64: 03's loop, one component at a time; rows with negligible responsibility are skipped."""
    N, d = X.shape
    R = np.zeros((N, K))
    R[np.arange(N), labels] = 1.0
    C = np.empty((K, d, d), complex)
    L = np.empty((N, K))
    hist, prev, converged, dropped, empty = [], -np.inf, False, 0.0, 0
    for _ in range(iters):
        Nk = R.sum(0)
        pi = (Nk + 1e-9) / N
        for k in range(K):
            idx = np.flatnonzero(R[:, k] > DROP)
            w = R[idx, k]
            S = np.zeros((d, d), complex)
            for i in range(0, len(idx), chunk):
                x = X[idx[i:i + chunk]]
                S += (x * w[i:i + chunk, None]).T @ x.conj()
            if Nk[k] > 1e-12:
                dropped = max(dropped, float(1 - w.sum() / Nk[k]))
                C[k] = (1 - alpha) * S / Nk[k] + alpha * Cg + eye
            else:                                      # empty component (03 would divide by zero)
                empty += 1
                C[k] = alpha * Cg + eye
            L[:, k] = np.log(pi[k]) + logpdf(X, C[k], chunk)
        ll, R = _posterior(L)
        cur = float(ll.mean())
        hist.append(round(float(cur), 4))
        if np.isfinite(prev) and abs(cur - prev) < tol * abs(prev):
            converged = True
            break
        prev = cur
    return pi, C, {"loglik": hist, "converged": converged, "hard": _hardness(L, ll),
                   "max_dropped_weight": float(f"{dropped:.3g}"), "empty_components": empty}


def _outer(x, buf, off):
    """[Re, Im] of conj(x_i) x_j for i <= j (row-major upper triangle) as one real row per vector."""
    n, d = x.shape
    o = buf[:n]
    xc = x.conj()
    for i in range(d):
        np.multiply(xc[:, i:i + 1], x[:, i:], out=o[:, off[i]:off[i + 1]])
    return o.view(np.float64)


def _em_features(X, K, alpha, labels, Cg, eye, iters, tol, chunk=512, snapshot_iter=20):
    """d <= 64: per chunk, L = logpi - logdet - F @ W gives all K log-densities and G += F^T R all K weighted
    covariances (one pass per iteration). F holds the upper triangle of x x^H; W is built so that F @ W equals
    03's full sum_ij conj(x_i) Ci_ij x_j for any Ci. Also keeps the model of iteration `snapshot_iter`
    (03's cap) when EM runs past it."""
    N, d = X.shape
    iu = np.triu_indices(d)
    off = np.r_[0, np.cumsum(np.arange(d, 0, -1))]
    diag = iu[0] == iu[1]
    buf = np.empty((chunk, len(iu[0])), complex)
    G = np.zeros((2 * len(iu[0]), K))
    for i in range(0, N, chunk):                       # M-step statistics of the k-means labels
        G += _outer(X[i:i + chunk], buf, off).T @ np.eye(K)[labels[i:i + chunk]]
    Nk = np.bincount(labels, minlength=K).astype(float)
    hist, prev, converged, empty, snap = [], -np.inf, False, 0, None
    hard = {"arg": np.empty(N, int), "maxpost": np.empty(N), "margin": np.empty(N)}
    for it in range(iters):
        pi = (Nk + 1e-9) / N
        T = np.ascontiguousarray(G.T).view(complex)   # (K, entries i<=j): sum_n r_nk conj(x_i) x_j
        S = np.empty((K, d, d), complex)               # 03's convention S_ij = sum_n r_nk x_i conj(x_j)
        S[:, iu[1], iu[0]] = T
        S[:, iu[0], iu[1]] = T.conj()
        ok = Nk > 1e-12
        empty = max(empty, int((~ok).sum()))
        C = (1 - alpha) * S / np.where(ok, Nk, 1.0)[:, None, None] + alpha * Cg + eye
        Ci = np.linalg.inv(C)
        Cu, Cl = Ci[:, iu[0], iu[1]], Ci[:, iu[1], iu[0]]
        W = np.empty((2 * len(iu[0]), K))
        W[0::2] = np.where(diag, Cu.real, Cu.real + Cl.real).T
        W[1::2] = np.where(diag, -Cu.imag, Cl.imag - Cu.imag).T
        base = np.log(pi) - np.linalg.slogdet(C)[1]
        if it + 1 == snapshot_iter:
            snap = (pi, C)
        G, Nk, tot = np.zeros_like(G), np.zeros(K), 0.0
        for i in range(0, N, chunk):
            F = _outer(X[i:i + chunk], buf, off)
            L = base - F @ W
            ll, R = _posterior(L)
            G += F.T @ R
            Nk += R.sum(0)
            tot += ll.sum()
            h = _hardness(L, ll)
            for key in hard:
                hard[key][i:i + len(L)] = h[key]
        cur = tot / N
        hist.append(round(float(cur), 4))
        if np.isfinite(prev) and abs(cur - prev) < tol * abs(prev):
            converged = True
            break
        prev = cur
    return pi, C, {"loglik": hist, "converged": converged, "hard": hard, "empty_components": empty,
                   "snapshot": snap if len(hist) > snapshot_iter else None}


def fit_em(X, K, alpha, labels, Cg, iters=ITERS, tol=TOL, heldout=None):
    """03's fit_gmm from given k-means labels, with diagnostics. Returns pi, C, info; info["snapshot"] is
    the model after 20 iterations (03's cap) when EM ran longer (d <= 64 only), else None."""
    t = time.time()
    d = X.shape[1]
    eye = 1e-6 * np.real(np.trace(Cg)) / d * np.eye(d)
    run = _em_features if d <= FEATURE_MAX_D else _em_direct
    pi, C, info = run(X, K, alpha, labels, Cg, eye, iters, tol)
    info.setdefault("snapshot", None)
    info["iters"] = len(info["loglik"])
    info["hit_cap"] = not info["converged"]
    info["train_posterior"] = _summary(info.pop("hard"), labels)
    if heldout is not None:
        Lh = np.stack([np.log(pi[k]) + logpdf(heldout, C[k]) for k in range(K)], 1)
        info["heldout_posterior"] = _summary(_hardness(Lh, _posterior(Lh)[0]))
    sizes = np.sort(pi * len(X))
    info["component_size_min_median_max"] = [round(float(sizes[0]), 1), round(float(np.median(sizes)), 1),
                                             round(float(sizes[-1]), 1)]
    info["model_sha256"] = digest(pi, C)
    info["fit_s"] = round(time.time() - t, 1)
    return pi, C, info


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:16]


def check_against_03():
    """fit_em must reproduce experiment 03's fit_gmm (both code paths) on a small random mixture."""
    rng = np.random.default_rng(7)
    out = {}
    for d in (24, 96):                                  # 24: feature path, 96: direct path
        A = rng.standard_normal((d, 10)) + 1j * rng.standard_normal((d, 10))
        X = np.concatenate([(rng.standard_normal((700, 10)) + 1j * rng.standard_normal((700, 10)))
                            @ (A * rng.random(10) ** 2).T for _ in range(4)])
        X += 0.1 * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
        with threadpool_limits(KMEANS_THREADS, user_api="openmp"):
            pi0, C0, info0 = fit_gmm(X, 4, 0.05, iters=8, seed=1, log=lambda s: None)
        Cg = cov(X)
        pi1, C1, info1 = fit_em(X, 4, 0.05, kmeans_labels(X, Cg, 4, 1), Cg, iters=8)
        dev = {"iters_03": info0["iters"], "iters_local": info1["iters"],
               "max_rel_dev_C": float(f"{np.max(np.abs(C1 - C0)) / np.max(np.abs(C0)):.2e}"),
               "max_dev_pi": float(f"{np.max(np.abs(pi1 - pi0)):.2e}"),
               "max_dev_loglik": float(f"{np.max(np.abs(np.subtract(info0['loglik'], info1['loglik']))):.2e}")}
        out["feature path (d=24)" if d <= FEATURE_MAX_D else "direct path (d=96)"] = dev
        if dev["iters_03"] != dev["iters_local"] or dev["max_rel_dev_C"] > 1e-6 or dev["max_dev_pi"] > 1e-6:
            raise RuntimeError(f"local EM does not reproduce experiment 03's fit_gmm: {dev}")
    return out


# ----------------------------------------------------------------------------- one setting
def run_setting(setting, a, log):
    t0 = time.time()
    D = load(a.data, setting)
    Xf, Xv, sel, rep, pats = D["Xf"], D["Xv"], D["sel"], D["rep"], D["pats"]
    del D["ff"], D["fv"], D["snap_f"]
    n_fit, d = Xf.shape
    grid = a.grid or (SMOKE_GRID if a.smoke else GRIDS[setting])
    em_max = min(a.em_max, {"spacefreq": 3000}.get(setting, 20_000)) if a.smoke else a.em_max
    iters = 3 if a.smoke else a.iters
    log(f"[{setting}] fit {Xf.shape}, val {Xv.shape} (select {int(sel.sum())}, report {int(rep.sum())}); "
        f"grid {grid}, seeds {a.seeds}, EM on {min(em_max, n_fit)} vectors, <= {iters} iterations")

    # LMMSE (K=1): global covariance of ALL training vectors, exactly as 03
    Cg_all = cov(Xf)
    Cg_all = Cg_all + 1e-6 * np.real(np.trace(Cg_all)) / d * np.eye(d)
    ev = np.sort(np.linalg.eigvalsh(Cg_all))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    eff_dim = {f"{p}%": int(np.searchsorted(cum, p / 100) + 1) for p in (90, 99, 99.9)}

    rb, rgid = np.unique(D["block"][rep], return_inverse=True)
    nrb = len(rb)
    brng = np.random.default_rng(0)                    # 03's draws: its EM subsample first, then the bootstrap
    brng.choice(n_fit, min(EM_MAX_03, n_fit), replace=False)
    boots = brng.integers(0, nrb, (B, nrb))
    Hs, Hr = Xv[sel], Xv[rep]
    withheld = {p: (np.setdiff1d(np.arange(d), P) if len(P) < d else np.arange(d)) for p, P in pats.items()}

    def noisy(pname, snr):                             # identical to 03: seed 7919 * len(P) + snr
        P = pats[pname]
        return add_noise(Xv[:, P], 10 ** (-snr / 10), seed=7919 * len(P) + snr)

    def nm(E, H, M):
        return float(10 * np.log10(np.sum(np.abs(E[:, M] - H[:, M]) ** 2) / np.sum(np.abs(H[:, M]) ** 2)))

    def per_block(E, M):
        return np.bincount(rgid, np.sum(np.abs(E[:, M] - Hr[:, M]) ** 2, 1), nrb)

    def bdb(err_b, ref_b):
        return 10 * np.log10(err_b[boots].sum(1) / ref_b[boots].sum(1))

    lmmse, lm_boot, ref_b = {}, {}, {}
    for pname, P in pats.items():
        M = withheld[pname]
        ref_b[pname] = np.bincount(rgid, np.sum(np.abs(Hr[:, M]) ** 2, 1), nrb)
        lmmse[pname] = {}
        for snr in SNRS:
            nv = 10 ** (-snr / 10)
            Y = noisy(pname, snr)
            Ys, Yr = Y[sel], Y[rep]
            score = {r: nm(mix_estimate(Ys, P, np.ones(1), Cg_all[None], nv, r), Hs, M) for r in RHOS}
            rho = min(RHOS, key=lambda r: score[r])
            E = mix_estimate(Yr, P, np.ones(1), Cg_all[None], nv, rho)
            lm_boot[pname, snr] = bdb(per_block(E, M), ref_b[pname])
            lmmse[pname][str(snr)] = {"nmse_db": round(nm(E, Hr, M), 3), "rho": rho,
                                      "select_nmse_db": round(score[rho], 3)}
    log(f"  LMMSE done ({time.time() - t0:.0f} s); effective dimension {eff_dim}")

    hidx = np.flatnonzero(sel)
    heldout = Xv[hidx[::max(1, len(hidx) // HELDOUT)][:HELDOUT]]
    Ks, alphas = sorted({k for k, _ in grid}), sorted({al for _, al in grid})
    per_seed = {}
    for seed in a.seeds:
        sub = (np.arange(n_fit) if em_max >= n_fit
               else np.sort(np.random.default_rng(seed).choice(n_fit, em_max, replace=False)))
        Xe = Xf if len(sub) == n_fit else Xf[sub]
        Cg = cov(Xe)
        labels, fits, scores, snaps = {}, {}, {}, {}
        for K, alpha in grid:
            name = f"K={K} a={alpha}"
            if K not in labels:
                t = time.time()
                labels[K] = kmeans_labels(Xe, Cg, K, seed)
                km_s = round(time.time() - t, 1)
            pi, C, info = fit_em(Xe, K, alpha, labels[K], Cg, iters=iters, heldout=heldout)
            snap = info.pop("snapshot")                     # d <= 64 only (a few MB): kept for the chosen rows
            if snap is not None:
                snaps[name] = snap
            info.update(kmeans_labels_sha256=digest(labels[K]), kmeans_s=km_s)
            fits[name] = info
            t = time.time()
            for pname, P in pats.items():
                M = withheld[pname]
                for snr in SNRS:
                    E = mix_estimate(noisy(pname, snr), P, pi, C, 10 ** (-snr / 10))   # rho = 0, as 03
                    scores[name, pname, snr] = (nm(E[sel], Hs, M), per_block(E[rep], M))
            del C, snap
            tp = info["train_posterior"]
            log(f"  seed {seed} {name:>12s}: {info['iters']:2d} it {'CAP' if info['hit_cap'] else 'conv'}, "
                f"loglik {info['loglik'][0]:.3f} -> {info['loglik'][-1]:.3f}, max post {tp['mean_max_posterior']:.3f}, "
                f"changed {tp['frac_changed_from_kmeans']:.3f}; fit {info['fit_s']:.0f} s + k-means {km_s:.0f} s "
                f"+ eval {time.time() - t:.0f} s  [{time.time() - t0:.0f} s]")
            km_s = 0.0
        rows = {}
        for pname in pats:
            rows[pname] = {}
            for snr in SNRS:
                names = [f"K={K} a={al}" for K, al in grid]
                best = min(names, key=lambda n: scores[n, pname, snr][0])
                K, alpha = grid[names.index(best)]
                err_b = scores[best, pname, snr][1]
                g = lm_boot[pname, snr] - bdb(err_b, ref_b[pname])
                nmse = round(float(10 * np.log10(err_b.sum() / ref_b[pname].sum())), 3)
                rep_all = {n: round(float(10 * np.log10(scores[n, pname, snr][1].sum() / ref_b[pname].sum())), 3)
                           for n in names}
                row = {
                    "chosen": best, "K": K, "alpha": alpha, "nmse_db": nmse,
                    "gain_over_lmmse_db": round(lmmse[pname][str(snr)]["nmse_db"] - nmse, 3),
                    "gain_ci95": [round(float(np.percentile(g, 2.5)), 3), round(float(np.percentile(g, 97.5)), 3)],
                    "at_largest_K": K == Ks[-1], "at_smallest_K": K == Ks[0],
                    "at_smallest_alpha": alpha == alphas[0], "at_largest_alpha": alpha == alphas[-1],
                    "select_nmse_db": {n: round(scores[n, pname, snr][0], 3) for n in names},
                    "report_nmse_db": rep_all}
                if best in snaps:                                   # did stopping at 20 iterations matter?
                    Y = noisy(pname, snr)[rep]
                    e20 = per_block(mix_estimate(Y, pats[pname], *snaps[best], 10 ** (-snr / 10)), withheld[pname])
                    row["nmse_db_at_iter20"] = round(float(10 * np.log10(e20.sum() / ref_b[pname].sum())), 3)
                    row["gain_from_iters_beyond_20_db"] = round(row["nmse_db_at_iter20"] - nmse, 3)
                rows[pname][str(snr)] = row
        per_seed[str(seed)] = {"em_vectors": int(len(sub)), "em_subsample_sha256": digest(sub),
                               "fits": fits, "rows": rows}
        del Xe

    aggregate = {}
    for pname in pats:
        aggregate[pname] = {}
        for snr in SNRS:
            r = [per_seed[str(s)]["rows"][pname][str(snr)] for s in a.seeds]
            gains = np.array([x["gain_over_lmmse_db"] for x in r])
            aggregate[pname][str(snr)] = {
                "lmmse_nmse_db": lmmse[pname][str(snr)]["nmse_db"],
                "gmm_nmse_db_mean": round(float(np.mean([x["nmse_db"] for x in r])), 3),
                "gain_db_mean": round(float(gains.mean()), 3),
                "gain_db_min": round(float(gains.min()), 3), "gain_db_max": round(float(gains.max()), 3),
                "gain_ci95_envelope": [min(x["gain_ci95"][0] for x in r), max(x["gain_ci95"][1] for x in r)],
                "seeds_with_ci_above_0": int(sum(x["gain_ci95"][0] > 0 for x in r)),
                "chosen_by_seed": {str(s): x["chosen"] for s, x in zip(a.seeds, r)},
                "seeds_at_largest_K": int(sum(x["at_largest_K"] for x in r)),
                "seeds_at_smallest_alpha": int(sum(x["at_smallest_alpha"] for x in r)),
                "seeds_at_corner": int(sum(x["at_largest_K"] and x["at_smallest_alpha"] for x in r)),
                "gain_from_iters_beyond_20_db_by_seed": {str(s): x.get("gain_from_iters_beyond_20_db")
                                                         for s, x in zip(a.seeds, r)}}
            ag = aggregate[pname][str(snr)]
            log(f"  {pname:>12s} {snr:2d} dB: LMMSE {ag['lmmse_nmse_db']:7.2f} | GMM gain mean {ag['gain_db_mean']:+.2f} "
                f"(seeds {ag['gain_db_min']:+.2f} .. {ag['gain_db_max']:+.2f}), CI envelope "
                f"[{ag['gain_ci95_envelope'][0]:+.2f}, {ag['gain_ci95_envelope'][1]:+.2f}], "
                f"chosen {sorted(set(ag['chosen_by_seed'].values()))}")
    fit_table = {f"K={K} a={al}": {
        "iters_by_seed": [per_seed[str(s)]["fits"][f"K={K} a={al}"]["iters"] for s in a.seeds],
        "hit_cap_by_seed": [per_seed[str(s)]["fits"][f"K={K} a={al}"]["hit_cap"] for s in a.seeds]}
        for K, al in grid}
    data = {"n_fit": int(n_fit), "dim": int(d), "n_select": int(sel.sum()), "n_report": int(rep.sum()),
            "report_blocks": int(nrb), "effective_dim": eff_dim, "scale": D["scale"]}
    metrics = {"data": data, "lmmse": lmmse, "aggregate": aggregate, "fits_summary": fit_table,
               "per_seed": per_seed, "runtime_s": round(time.time() - t0, 1)}
    config = {"setting": setting, "grid": [list(c) for c in grid], "seeds": a.seeds, "em_max": em_max,
              "em_iters_cap": iters, "em_tol": TOL, "em_stop_rule": "|ll_t - ll_t-1| < tol |ll_t-1| (03)",
              "bootstrap_B": B, "kmeans_threads": KMEANS_THREADS, "drop_threshold": DROP,
              "feature_path_max_d": FEATURE_MAX_D, "heldout_diagnostic_vectors": int(len(heldout)),
              "rho_grid": RHOS, "snrs_db": SNRS, "patterns": {p: P.tolist() for p, P in pats.items()},
              "noise_seed": "7919 * len(P) + snr (as 03)", "gmm_rho": 0.0,
              "fit_runs": [53, 54, 55], "val_run": 56, "threshold_db": -30.0, "block_len": 50,
              "models_saved": False,
              "model_note": "covariances not saved (too large); refit with the same data, seed, code and "
                            "OPENBLAS_NUM_THREADS and compare model_sha256", "smoke": a.smoke,
              "synthetic": bool(a.synthetic)}
    return config, metrics


# ----------------------------------------------------------------------------- figures
TITLES = {"32ant": "Pilots on 32 of 32 antennas", "16ant": "Pilots on 16 of 32 antennas",
          "8ant": "Pilots on 8 of 32 antennas", "32ant x 4sc": "Pilots: 32 antennas x 4 subcarriers",
          "8ant x 4sc": "Pilots: 8 antennas x 4 subcarriers"}


def _footer(fig, axes, ncol, note):
    """Shared legend and a one-line note in a reserved strip under the panels."""
    import matplotlib.pyplot as plt
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.5, 0.085), ncol=ncol,
               frameon=False)
    fig.text(0.5, 0.0, note, ha="center", va="bottom", fontsize=plt.rcParams["font.size"] - 1, color="#52514e")


def plot_gain(metrics, seeds, path):
    from style import COLORS, LABELS, MARKERS, PAGE_W, apply_style
    plt = apply_style("paper")
    agg = metrics["aggregate"]
    pats = list(agg)
    fig, axes = plt.subplots(1, len(pats), figsize=(PAGE_W, 2.3), squeeze=False)
    for ax, pname in zip(axes[0], pats):
        rows = [agg[pname][str(s)] for s in SNRS]
        mean = np.array([r["gain_db_mean"] for r in rows])
        lo = np.array([r["gain_ci95_envelope"][0] for r in rows])
        hi = np.array([r["gain_ci95_envelope"][1] for r in rows])
        ax.axhline(0, color=COLORS["lmmse"], linewidth=1.0, label=LABELS["lmmse"])
        for j, s in enumerate(seeds):
            g = [metrics["per_seed"][str(s)]["rows"][pname][str(x)]["gain_over_lmmse_db"] for x in SNRS]
            ax.plot(SNRS, g, linestyle="none", marker="o", markersize=2.5, markerfacecolor="none",
                    markeredgecolor=COLORS["genie"], label="Single seeds" if j == 0 else None)
        ax.errorbar(SNRS, mean, yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], color=COLORS["gmm"],
                    marker=MARKERS["gmm"], capsize=2, elinewidth=0.8,
                    label=f"{LABELS['gmm']}, mean of {len(seeds)} seeds")
        ax.set_title(TITLES.get(pname, pname), loc="left")
        ax.set_xlabel("Pilot SNR (dB)")
        ax.set_xticks(SNRS)
    axes[0][0].set_ylabel("Gain over LMMSE (dB)")
    _footer(fig, axes, 3, "Error bars: envelope of the per-seed paired 95% block-bootstrap CIs "
            "(validation run 0056, odd blocks)")
    fig.savefig(path)
    plt.close(fig)


def plot_grid(metrics, config, seeds, path, snr=0):
    """Selection-block (even) gain over LMMSE at one SNR, per (K, alpha): is the chosen K inside the grid?"""
    from style import COLORS, MARKERS, PAGE_W, apply_style
    plt = apply_style("paper")
    grid = [tuple(c) for c in config["grid"]]
    alphas = sorted({al for _, al in grid})
    pats = list(metrics["aggregate"])
    fig, axes = plt.subplots(1, len(pats), figsize=(PAGE_W, 2.3), squeeze=False)
    for ax, pname in zip(axes[0], pats):
        lm = metrics["lmmse"][pname][str(snr)]["select_nmse_db"]
        for al in alphas:
            Ks = sorted(K for K, a_ in grid if a_ == al)
            y = [np.mean([lm - metrics["per_seed"][str(s)]["rows"][pname][str(snr)]["select_nmse_db"][f"K={K} a={al}"]
                          for s in seeds]) for K in Ks]
            ax.plot(Ks, y, color=COLORS["gmm"], marker=MARKERS["gmm"], linestyle="-" if al == alphas[-1] else "--",
                    label=rf"shrinkage $\alpha$ = {al}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({K for K, _ in grid}), [str(K) for K in sorted({K for K, _ in grid})])
        ax.minorticks_off()
        ax.set_title(TITLES.get(pname, pname), loc="left")
        ax.set_xlabel("Mixture components K")
    axes[0][0].set_ylabel("Gain over LMMSE (dB)")
    _footer(fig, axes, len(alphas), f"Pilot SNR {snr} dB; even (selection) blocks of validation run 0056; "
            "mean over seeds")
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- synthetic test hook
def synthetic_npz(path, n_snap, seed=0):
    """A small file with the keys and shapes of the processed file, from a ray-like model: a transmitter
    moving through a 6 m x 4 m room, 8 local scatterers, 32-antenna ULA, 64 subcarriers of 48.8 kHz.
    Runs 0053-0056 only. For code tests; its numbers mean nothing."""
    rng = np.random.default_rng(seed)
    per = n_snap // 4
    run = np.repeat([53, 54, 55, 56], per)
    index = np.tile(np.arange(per), 4)
    ph = rng.uniform(0, 2 * np.pi, (4, 2)).repeat(per, 0)
    pos = np.column_stack([3 + 2.5 * np.sin(0.011 * index + ph[:, 0]), 2 + 1.6 * np.sin(0.017 * index + ph[:, 1])])
    rot = np.sin(0.005 * index + ph[:, 0])
    scat = np.column_stack([rng.uniform(0, 6, 8), rng.uniform(0, 4, 8)])
    arr = np.array([3.0, -1.0])
    f = (np.arange(64) - 32) * 50e6 / 1024
    ant = np.arange(32)
    csi = np.zeros((len(run), 32, 64), np.complex64)
    for t in range(len(run)):
        u = pos[t]
        pts = np.vstack([u, scat])
        dist = np.linalg.norm(u - pts, axis=1) + np.linalg.norm(pts - arr, axis=1)
        amp = np.r_[0.3, np.exp(-np.sum((scat - u) ** 2, 1) / 2.0)] / dist
        theta = np.arctan2(pts[:, 0] - arr[0], pts[:, 1] - arr[1])
        theta = (theta[:, None] + np.radians([-2, 0, 2])[None]).ravel()
        g = (np.repeat(amp * np.exp(2j * np.pi * dist / 0.24), 3) / np.sqrt(3)
             * (rng.standard_normal(len(theta)) + 1j * rng.standard_normal(len(theta))) / np.sqrt(2))
        tau = np.repeat(dist / 3e8, 3)
        H = (np.exp(1j * np.pi * ant[:, None] * np.sin(theta)[None]) * g) @ np.exp(-2j * np.pi * tau[:, None] * f)
        csi[t] = H * np.exp(1j * rng.uniform(0, 2 * np.pi))
    p = np.mean(np.abs(csi) ** 2)
    csi += np.sqrt(p * 10 ** -3.5 / 2) * (rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape))
    quality = np.full((len(run), 32), -40.0, np.float32)
    quality[rng.random(quality.shape) < 0.002] = -20.0                   # a few screened vectors
    np.savez(path, csi=csi, quality=quality, run=run, index=index, pos=pos, rot=rot, block=np.array([256, 320]))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--data", default=DEFAULT_NPZ)
    ap.add_argument("--settings", nargs="+", default=["spacefreq", "space32"], choices=list(GRIDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--em-max", type=int, default=EM_MAX_03, help="max training vectors for EM (03: 200000)")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--grid", nargs="+", default=None, metavar="K:ALPHA", help="override the grid, e.g. 16:0.05")
    ap.add_argument("--smoke", action="store_true", help="code test on a small part of the data; not saved")
    ap.add_argument("--synthetic", type=int, default=0, metavar="N_SNAP",
                    help="test hook: run on a synthetic file of N_SNAP snapshots (never the real data); not saved")
    a = ap.parse_args()
    a.grid = [(int(g.split(":")[0]), float(g.split(":")[1])) for g in a.grid] if a.grid else None
    log = lambda m: print(m, flush=True)
    out_dir = tempfile.mkdtemp(prefix="gmm_seeds_")
    if a.synthetic:
        a.data = os.path.join(out_dir, "synthetic.npz")
        synthetic_npz(a.data, a.synthetic)
    check = check_against_03()
    log(f"local EM vs 03's fit_gmm: {check}")
    for setting in a.settings:
        config, metrics = run_setting(setting, a, log)
        try:
            config["data"] = os.path.relpath(a.data, ROOT).replace(os.sep, "/")
        except ValueError:                                                  # different drive
            config["data"] = os.path.basename(a.data)
        metrics["equivalence_check_vs_03"] = check
        files = []
        for name, fn in (("gain_vs_snr", lambda p: plot_gain(metrics, a.seeds, p)),
                         ("grid_check_0dB", lambda p: plot_grid(metrics, config, a.seeds, p))):
            path = os.path.join(out_dir, f"{setting}_{name}.png")
            try:                                       # a plotting problem must never lose the metrics
                fn(path)
                files.append(path)
            except Exception as e:  # noqa: BLE001
                log(f"plot {name} failed ({e!r}); metrics are still saved")
        if a.smoke or a.synthetic:
            path = os.path.join(out_dir, f"{setting}_metrics.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"config": config, "metrics": metrics}, f, indent=1, default=str)
            log(f"{'smoke' if a.smoke else 'synthetic'} test of {setting} finished (nothing saved to results/); "
                f"outputs in {out_dir}")
            continue
        save_run(f"gmm-seeds-{setting}", a.member, config, metrics, files=files,
                 note=f"GMM vs LMMSE ({setting}), seeds {a.seeds}, validation run 0056 (odd 50-snapshot blocks "
                      "reported, even blocks for every selection); test run 0057 untouched; GMM covariances not "
                      "saved, refits are deterministic from the seed (model_sha256)")
    if a.synthetic:
        os.remove(a.data)                              # the synthetic input file is only a test fixture


if __name__ == "__main__":
    main()
