"""Feasibility check (2026-11-19): can a Gaussian-mixture estimator beat global-covariance LMMSE?

Method (Turan, Fesl, Grundei, Koller, Utschick, arXiv 2207.14150): fit a K-component Gaussian
mixture to clean training channels by EM; for noisy pilots y = h_P + n the estimate is
    h_hat = sum_k p(k|y) C_k[:,P] (C_k[P,P] + s2 I)^-1 y,
    p(k|y) ~ pi_k CN(y; 0, C_k[P,P] + s2 I).
K = 1 is the global-covariance LMMSE baseline. Components are zero-mean because the
calibrated CSI has a random common phase per snapshot (|mean|^2 / power ~ 5e-5).

Diagnostic reference "cell-genie" (NOT a usable receiver): one covariance per spatial cell of the
transmitter position (k-means on training positions; variant "cellrot" also uses orientation), and
the TRUE test position picks the cell. It indicates how much a position-aware mixture could gain.

Settings
  freq64     one antenna, 64 subcarriers (the current task); pilots on 16 / 8 / 4 subcarriers
  space32    one subcarrier, 32 antennas (8 subcarriers spread over the block used as samples);
             pilots on all 32 antennas (pure denoising) / 16 / 8 antennas (index stride 2 / 4)
  spacefreq  32 antennas x 16 subcarriers (every ~4th of the 64-block) = 512 entries;
             pilots on 32 ant x 4 sc and 8 ant x 4 sc

Protocol as experiments/01: fit on runs 0053-0055; validation run 0056 cut into 50-snapshot
blocks, even blocks choose hyper-parameters, odd blocks are reported; test run 0057 is never
loaded; one global scale; pilot SNR relative to the average channel power; NMSE on the withheld
entries (on all entries when every entry is a pilot). Same noise realisation for every method.

    python experiments/03_gmm_feasibility/gmm_feasibility.py --setting space32
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from estimators import add_noise, pilot_positions  # noqa: E402

SNRS = [0, 10, 20, 30]
RHOS = [0.0, 1e-4, 1e-3, 1e-2]          # LMMSE diagonal loading (x mean pilot power)
CONFIGS = {                              # (K, alpha) for the GMM; alpha = shrinkage to global cov
    "freq64": [(4, 0.05), (16, 0.05), (64, 0.05)],
    "space32": [(4, 0.05), (16, 0.05), (64, 0.05)],
    "spacefreq": [(4, 0.05), (16, 0.05), (4, 0.3), (16, 0.3)],
}
CELLS = [(16, 0.05), (64, 0.05), (16, 0.3), (64, 0.3)]   # (number of cells, alpha)


# ----------------------------------------------------------------------------- data
def load(path, setting, thr=-30.0, block_len=50):
    z = np.load(path)
    csi, q, run, index, pos, rot = z["csi"], z["quality"], z["run"], z["index"], z["pos"], z["rot"]
    fit_s, val_s = np.isin(run, (53, 54, 55)), run == 56
    feat = np.column_stack([pos, np.cos(rot), np.sin(rot)])          # x, y (m), orientation
    if setting == "freq64":                                          # per (snapshot, antenna)
        keep = q <= thr

        def vec(m):
            mm = m[:, None] & keep
            sid = np.broadcast_to(np.arange(len(run))[:, None], keep.shape)[mm]
            return csi[mm].astype(np.complex128), sid
        Xf, sf = vec(fit_s)
        Xv, sv = vec(val_s)
        pats = {f"{n}sc": pilot_positions(64, n) for n in (16, 8, 4)}
    else:                                                            # snapshots with all antennas ok
        good = (q <= thr).all(1)
        if setting == "space32":
            sc = pilot_positions(64, 8)

            def vec(m):
                idx = np.flatnonzero(m & good)
                X = csi[idx][:, :, sc].transpose(0, 2, 1).reshape(-1, 32)
                return X.astype(np.complex128), np.repeat(idx, len(sc))
            pats = {"32ant": np.arange(32), "16ant": np.arange(0, 32, 2), "8ant": np.arange(0, 32, 4)}
        elif setting == "spacefreq":
            sc = pilot_positions(64, 16)
            sc_p = pilot_positions(16, 4)                            # 4 of the 16 grid subcarriers

            def vec(m):
                idx = np.flatnonzero(m & good)
                return csi[idx][:, :, sc].reshape(len(idx), -1).astype(np.complex128), idx
            grid = lambda ants: (ants[:, None] * 16 + sc_p[None, :]).ravel()   # entry = ant*16 + sc
            pats = {"32ant x 4sc": grid(np.arange(32)), "8ant x 4sc": grid(np.arange(0, 32, 4))}
        else:
            raise ValueError(setting)
        Xf, sf = vec(fit_s)
        Xv, sv = vec(val_s)
    scale = 1 / np.sqrt(np.mean(np.abs(Xf) ** 2))
    block = index[sv] // block_len
    return dict(Xf=Xf * scale, Xv=Xv * scale, ff=feat[sf], fv=feat[sv], snap_f=sf,
                sel=block % 2 == 0, rep=block % 2 == 1, block=block, pats=pats, scale=float(scale))


# ----------------------------------------------------------------------------- GMM
def cov(X, w=None, chunk=50_000):
    d = X.shape[1]
    C = np.zeros((d, d), complex)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        C += (x if w is None else x * w[i:i + chunk, None]).T @ x.conj()
    return C / (len(X) if w is None else w.sum())


def logpdf(X, C, chunk=50_000):
    """log CN(x; 0, C) for the rows of X, without the constant -d log(pi)."""
    Ci = np.linalg.inv(C)
    logdet = np.linalg.slogdet(C)[1]
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        out[i:i + chunk] = -logdet - np.sum((x.conj() @ Ci) * x, 1).real
    return out


def fit_gmm(X, K, alpha, iters=20, seed=0, tol=1e-5, log=print):
    N, d = X.shape
    Cg = cov(X)
    eye = 1e-6 * np.real(np.trace(Cg)) / d * np.eye(d)
    if K == 1:
        return np.ones(1), (Cg + eye)[None], {"iters": 0}
    # initialisation: k-means on unit-norm vectors with the common phase removed
    v = np.linalg.eigh(Cg)[1][:, -1]
    F = X * np.exp(-1j * np.angle(X @ v.conj()))[:, None] / np.linalg.norm(X, axis=1)[:, None]
    lab = KMeans(K, n_init=1, random_state=seed).fit_predict(np.hstack([F.real, F.imag]).astype(np.float32))
    R = np.eye(K)[lab]
    prev = -np.inf
    hist = []
    for it in range(iters):
        Nk = R.sum(0) + 1e-9
        pi = Nk / N
        C = np.stack([(1 - alpha) * cov(X, R[:, k]) + alpha * Cg + eye for k in range(K)])
        L = np.stack([np.log(pi[k]) + logpdf(X, C[k]) for k in range(K)], 1)
        m = L.max(1, keepdims=True)
        ll = m[:, 0] + np.log(np.exp(L - m).sum(1))
        R = np.exp(L - ll[:, None])
        cur = float(ll.mean())
        hist.append(round(cur, 4))
        if np.isfinite(prev) and abs(cur - prev) < tol * abs(prev):
            break
        prev = cur
    log(f"    EM K={K} alpha={alpha}: {len(hist)} iters, mean loglik {hist[0]:.3f} -> {hist[-1]:.3f}")
    return pi, C, {"iters": len(hist), "loglik": hist}


def mix_estimate(Y, P, pi, C, nv, rho=0.0):
    """sum_k p(k|y) C_k[:,P] (C_k[P,P] + s2 I)^-1 y  (K = 1: plain LMMSE)."""
    m = len(P)
    ppow = float(np.real(np.trace(C[0][np.ix_(P, P)])) / m)
    S = [C[k][np.ix_(P, P)] + (nv + rho * ppow) * np.eye(m) for k in range(len(pi))]
    G = [C[k][:, P] @ np.linalg.inv(S[k]) for k in range(len(pi))]
    if len(pi) == 1:
        return Y @ G[0].T
    L = np.stack([np.log(pi[k]) + logpdf(Y, S[k]) for k in range(len(pi))], 1)
    W = np.exp(L - L.max(1, keepdims=True))
    W /= W.sum(1, keepdims=True)
    est = np.zeros((len(Y), C.shape[1]), complex)
    for k in range(len(pi)):
        est += W[:, k, None] * (Y @ G[k].T)
    return est


def cell_estimate(Y, P, Cc, lab, nv):
    est = np.zeros((len(Y), Cc.shape[1]), complex)
    for c in np.unique(lab):
        i = lab == c
        est[i] = mix_estimate(Y[i], P, np.ones(1), Cc[c][None], nv)
    return est


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=list(CONFIGS))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "processed", "dichasus005x_sc256-319.npz"))
    ap.add_argument("--em-max", type=int, default=200_000, help="max training vectors for EM / cells")
    ap.add_argument("--smoke", action="store_true", help="tiny subsample, one config: code test only")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    t0 = time.time()
    logf = open(os.path.join(a.out, f"log_{a.setting}{'_smoke' if a.smoke else ''}.txt"), "w", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    D = load(a.data, a.setting)
    Xf, Xv, sel, rep = D["Xf"], D["Xv"], D["sel"], D["rep"]
    rng = np.random.default_rng(0)
    em_max = 20_000 if a.smoke else a.em_max
    sub = np.sort(rng.choice(len(Xf), min(em_max, len(Xf)), replace=False))
    configs = CONFIGS[a.setting][:1] if a.smoke else CONFIGS[a.setting]
    cells = CELLS[:1] if a.smoke else CELLS
    log(f"[{a.setting}] fit {Xf.shape} (EM on {len(sub)}), val {Xv.shape}, select {sel.sum()}, report {rep.sum()}")

    # global LMMSE uses ALL training vectors; GMM / cells use the EM subsample (conservative for them)
    Cg = cov(Xf)
    d = Xf.shape[1]
    Cg = Cg + 1e-6 * np.real(np.trace(Cg)) / d * np.eye(d)
    ev = np.sort(np.linalg.eigvalsh(Cg))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    eff_dim = {f"{p}%": int(np.searchsorted(cum, p / 100) + 1) for p in (90, 99, 99.9)}
    log(f"  effective linear dimension (eigenvalues for 90/99/99.9% energy): {eff_dim}")

    models = {}
    for K, alpha in configs:
        t = time.time()
        pi, C, info = fit_gmm(Xf[sub], K, alpha, iters=5 if a.smoke else 20, log=log)
        models[f"gmm K={K} a={alpha}"] = (pi, C, dict(info, fit_s=round(time.time() - t, 1)))
    feat_names = {"cell": slice(0, 2), "cellrot": slice(0, 4)}
    cellmods = {}
    Xs, fs = Xf[sub], D["ff"][sub]
    _, first = np.unique(D["snap_f"], return_index=True)     # each training snapshot counted once
    for kind, sl in feat_names.items():
        for Kc in sorted({c[0] for c in cells}):
            km = KMeans(Kc, n_init=3, random_state=0).fit(D["ff"][first][:, sl])
            lab_f, lab_v = km.predict(fs[:, sl]), km.predict(D["fv"][:, sl])
            raw = [cov(Xs[lab_f == c]) if np.sum(lab_f == c) > 1 else Cg for c in range(Kc)]
            for alpha in sorted({c[1] for c in cells if c[0] == Kc}):
                Cc = np.stack([(1 - alpha) * R_ + alpha * Cg for R_ in raw])
                cellmods[f"{kind}-genie N={Kc} a={alpha}"] = (Cc, lab_v)
    del Xs

    # paired block bootstrap on the reported (odd) blocks
    rb, rgid = np.unique(D["block"][rep], return_inverse=True)
    boots = rng.integers(0, len(rb), (1000, len(rb)))

    def per_block(E, H, M):
        return np.bincount(rgid, np.sum(np.abs(E[:, M] - H[:, M]) ** 2, 1), len(rb))

    results = {}
    for pname, P in D["pats"].items():
        M = np.setdiff1d(np.arange(d), P)
        M = M if len(M) else np.arange(d)
        results[pname] = {}
        for snr in SNRS:
            nv = 10 ** (-snr / 10)
            Y = add_noise(Xv[:, P], nv, seed=7919 * len(P) + snr)
            Hs, Hr, Ys, Yr = Xv[sel], Xv[rep], Y[sel], Y[rep]
            nm = lambda E, H: float(10 * np.log10(np.sum(np.abs(E[:, M] - H[:, M]) ** 2) / np.sum(np.abs(H[:, M]) ** 2)))
            row, sel_scores = {}, {}
            # LMMSE: choose loading on even blocks
            rho = min(RHOS, key=lambda r: nm(mix_estimate(Ys, P, np.ones(1), Cg[None], nv, r), Hs))
            E_l = mix_estimate(Yr, P, np.ones(1), Cg[None], nv, rho)
            row["lmmse"] = {"nmse_db": round(nm(E_l, Hr), 3), "rho": rho}
            ref_b = np.bincount(rgid, np.sum(np.abs(Hr[:, M]) ** 2, 1), len(rb))
            bl = 10 * np.log10(per_block(E_l, Hr, M)[boots].sum(1) / ref_b[boots].sum(1))
            for fam, cand in (("gmm", models), ("cell-genie", {k: v for k, v in cellmods.items() if k.startswith("cell-")}),
                              ("cellrot-genie", {k: v for k, v in cellmods.items() if k.startswith("cellrot")})):
                allrep = {}
                for name, mod in cand.items():
                    if fam == "gmm":
                        es = mix_estimate(Ys, P, mod[0], mod[1], nv)
                        er = mix_estimate(Yr, P, mod[0], mod[1], nv)
                    else:
                        es = cell_estimate(Ys, P, mod[0], mod[1][sel], nv)
                        er = cell_estimate(Yr, P, mod[0], mod[1][rep], nv)
                    sel_scores[name] = nm(es, Hs)
                    allrep[name] = (nm(er, Hr), er)
                best = min(cand, key=lambda k: sel_scores[k])
                b = 10 * np.log10(per_block(allrep[best][1], Hr, M)[boots].sum(1) / ref_b[boots].sum(1))
                g = bl - b
                row[fam] = {"chosen": best, "nmse_db": round(allrep[best][0], 3),
                            "gain_over_lmmse_db": round(row["lmmse"]["nmse_db"] - allrep[best][0], 3),
                            "gain_ci95": [round(float(np.percentile(g, 2.5)), 3), round(float(np.percentile(g, 97.5)), 3)],
                            "all_report_nmse_db": {k: round(v[0], 3) for k, v in allrep.items()}}
            results[pname][str(snr)] = row
            log(f"  {pname:>12s} {snr:2d} dB: LMMSE {row['lmmse']['nmse_db']:7.2f} | "
                + " | ".join(f"{f} {row[f]['gain_over_lmmse_db']:+.2f} [{row[f]['gain_ci95'][0]:+.2f},{row[f]['gain_ci95'][1]:+.2f}]"
                             for f in ("gmm", "cell-genie", "cellrot-genie")))

    out = {"setting": a.setting, "smoke": a.smoke, "scale": D["scale"], "dim": d,
           "n_fit": int(len(Xf)), "n_em": int(len(sub)), "n_select": int(sel.sum()), "n_report": int(rep.sum()),
           "report_blocks": int(len(rb)), "effective_dim": eff_dim, "snrs_db": SNRS,
           "gmm_fit": {k: v[2] for k, v in models.items()}, "results": results,
           "runtime_s": round(time.time() - t0, 1),
           "note": "validation run 0056 only (odd 50-snapshot blocks reported, even blocks for selection); "
                   "test run 0057 untouched; gains are dB better than global-covariance LMMSE, paired block-bootstrap 95% CI; "
                   "cell-genie uses the true test position and is NOT a usable estimator"}
    with open(os.path.join(a.out, f"results_{a.setting}{'_smoke' if a.smoke else ''}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
