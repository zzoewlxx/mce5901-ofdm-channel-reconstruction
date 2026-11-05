"""Rigorous re-check of the reformulated study (noisy pilots), on the cached sample.

Same sample as pilot_check.py (no new downloads):
  fit   = run 0053, 40 chunks x 3 snapshots (all 32 antennas pooled)
  val   = run 0053, 10 interleaved chunks (hyper-parameters / early stopping)
  eval  = run 0056, 30 chunks x 3 snapshots
Run 0057 (the proposal's test run) is not touched.

Checks
  1  is the error of the measured CSI white?  (three noise estimators)
  2  noisy pilots: two SNR conventions, 6 blocks, 16/8/4 of 64 pilots,
     linear / spline / LMMSE (training statistics) / LMMSE (genie statistics of eval run)
  3  bootstrap 95% CIs over evaluation chunks
  4  spread over the 32 antennas
  5  LMMSE versus amount of training data
  6  spatial shift (fit on one part of the room)
  7  numpy MLP (2x128 ReLU, Adam), trained over pilot SNR U[0,30] dB, 3 seeds,
     plus training-size effect

Usage:  python -I rigor_check.py --cache DIR [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_check import (calibrate, linear_matrix, load_offsets, load_sample,  # noqa: E402
                         pilot_positions, spline_matrix)

SNRS = [0, 5, 10, 15, 20, 25, 30]
RHOS = [1e-4, 1e-3, 1e-2, 1e-1]


def db(x):
    return float(10 * np.log10(x))


def cn(rng, shape):
    """Unit-variance circular complex Gaussian."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def nmse(est, ref):
    return db(np.sum(np.abs(est - ref) ** 2) / np.sum(np.abs(ref) ** 2))


def sums_by(est, ref, groups, n_groups):
    e = np.sum(np.abs(est - ref) ** 2, axis=1)
    r = np.sum(np.abs(ref) ** 2, axis=1)
    return np.bincount(groups, e, n_groups), np.bincount(groups, r, n_groups)


# ------------------------------------------------------------------ LMMSE
def stats(H):
    mu = H.mean(0)
    X = H - mu
    return mu, (X.T @ X.conj()) / len(X)


def lmmse_G(C, p, noise_var, rho):
    ppow = np.real(np.trace(C[np.ix_(p, p)])) / len(p)
    Cpp = C[np.ix_(p, p)] + (noise_var + rho * ppow) * np.eye(len(p))
    return np.linalg.solve(Cpp.T, C[:, p].T).T


def lmmse_apply(mu, G, Y, p):
    return mu + (Y - mu[p]) @ G.T


# ------------------------------------------------------------------- MLP
class MLP:
    def __init__(self, d_in, d_out, hidden, rng):
        def he(a, b, gain=2.0):
            return rng.standard_normal((a, b)) * np.sqrt(gain / a)
        self.P = [he(d_in, hidden), np.zeros(hidden), he(hidden, hidden), np.zeros(hidden),
                  he(hidden, d_out, 1.0), np.zeros(d_out)]
        self.m = [np.zeros_like(w) for w in self.P]
        self.v = [np.zeros_like(w) for w in self.P]
        self.t = 0

    def n_params(self):
        return int(sum(w.size for w in self.P))

    def forward(self, X):
        W1, b1, W2, b2, W3, b3 = self.P
        z1 = X @ W1 + b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ W2 + b2
        h2 = np.maximum(z2, 0)
        return h2 @ W3 + b3, (X, z1, h1, z2, h2)

    def step(self, cache, dout, lr, b1=0.9, b2=0.999, eps=1e-8):
        X, z1, h1, z2, h2 = cache
        W1, _, W2, _, W3, _ = self.P
        dz2 = (dout @ W3.T) * (z2 > 0)
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        grads = [X.T @ dz1, dz1.sum(0), h1.T @ dz2, dz2.sum(0), h2.T @ dout, dout.sum(0)]
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            self.P[i] -= lr * mh / (np.sqrt(vh) + eps)


def mlp_features(Y, snr_db):
    return np.concatenate([Y.real, Y.imag, (np.asarray(snr_db, float) / 30.0)[:, None]], axis=1)


def mlp_predict(net, Y, snr_db, n_miss):
    out, _ = net.forward(mlp_features(Y, np.full(len(Y), snr_db)))
    return out[:, :n_miss] + 1j * out[:, n_miss:]


def train_mlp(Hf, Hv, p, miss, seed, epochs=160, batch=128, hidden=128):
    rng = np.random.default_rng(seed)
    net = MLP(2 * len(p) + 1, 2 * len(miss), hidden, rng)
    val_snrs = [0, 10, 20, 30]
    Zv = {s: cn(np.random.default_rng(5000 + s), (len(Hv), len(p))) for s in val_snrs}
    best, best_P, t0 = np.inf, None, time.time()
    for ep in range(epochs):
        lr = 1e-3 if ep < 0.6 * epochs else (3e-4 if ep < 0.85 * epochs else 1e-4)
        idx = rng.permutation(len(Hf))
        for b in range(0, len(idx), batch):
            h = Hf[idx[b:b + batch]]
            s = rng.uniform(0, 30, len(h))
            Y = h[:, p] + np.sqrt(10 ** (-s / 10))[:, None] * cn(rng, (len(h), len(p)))
            out, cache = net.forward(mlp_features(Y, s))
            T = np.concatenate([h[:, miss].real, h[:, miss].imag], axis=1)
            net.step(cache, 2 * (out - T) / out.size, lr)
        if ep % 4 == 3 or ep == epochs - 1:
            v = np.mean([nmse(mlp_predict(net, Hv[:, p] + np.sqrt(10 ** (-s / 10)) * Zv[s], s,
                                          len(miss)), Hv[:, miss]) for s in val_snrs])
            if v < best:
                best, best_P = v, [w.copy() for w in net.P]
    net.P = best_P
    return net, best, time.time() - t0


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--quick", action="store_true", help="fewer MLP runs")
    a = ap.parse_args()
    t_start = time.time()
    res = {}

    tr = load_sample("0053", 50, 3, a.cache)
    ev = load_sample("0056", 30, 3, a.cache)
    sto, cpo = load_offsets(a.cache)
    tr["csi"], ev["csi"] = calibrate(tr["csi"], sto, cpo), calibrate(ev["csi"], sto, cpo)
    n_ant = tr["csi"].shape[1]
    fit_s = (tr["chunk"] % 5) != 4
    val_s = ~fit_s
    # per-vector bookkeeping (vectors are snapshot-major, antenna-minor)
    ev_chunk = np.repeat(ev["chunk"], n_ant)
    ev_ant = np.tile(np.arange(n_ant), len(ev["csi"]))
    ev_x = np.repeat(ev["pos"][:, 0], n_ant)
    tr_chunk = np.repeat(tr["chunk"], n_ant)
    tr_x = np.repeat(tr["pos"][:, 0], n_ant)
    fit_v, val_v = np.repeat(fit_s, n_ant), np.repeat(val_s, n_ant)
    n_eval_chunks = int(ev["chunk"].max() + 1)

    # ------------------------------------------------ 1. noise structure
    allc = np.concatenate([tr["csi"], ev["csi"]]).reshape(-1, 1024).astype(np.complex128)
    g = np.fft.ifft(np.fft.ifftshift(allc, axes=-1), axis=-1, norm="ortho")
    s_dd = np.mean(np.abs(g[:, 256:768]) ** 2, axis=1)
    d4, nb = [], []
    for lo, hi in [(130, 508), (517, 892)]:
        k = np.arange(lo, hi)
        d4.append(allc[:, k - 2] - 4 * allc[:, k - 1] + 6 * allc[:, k] - 4 * allc[:, k + 1] + allc[:, k + 2])
        nb.append(allc[:, k] - 0.5 * (allc[:, k - 1] + allc[:, k + 1]))
    s_4d = np.mean(np.abs(np.concatenate(d4, 1)) ** 2, axis=1) / 70.0
    s_nb = np.mean(np.abs(np.concatenate(nb, 1)) ** 2, axis=1) / 1.5
    band = np.r_[128:510, 515:894]
    sig = np.mean(np.abs(allc[:, band]) ** 2, axis=1)
    res["noise"] = {name: {"pooled_db": db(np.sum(s) / np.sum(sig)),
                           "median_db": float(np.median(10 * np.log10(s / sig)))}
                    for name, s in [("delay_domain", s_dd), ("fourth_difference", s_4d),
                                    ("neighbour_interp", s_nb)]}
    print("[1] error level of the measured CSI relative to signal power (dB):")
    for k, v in res["noise"].items():
        print(f"    {k:18s} pooled {v['pooled_db']:6.1f}   median {v['median_db']:6.1f}")

    # ------------------------------------------------ helpers per block
    def block(sl, normalize):
        Ht = tr["csi"][:, :, sl].reshape(-1, sl.stop - sl.start).astype(np.complex128)
        He = ev["csi"][:, :, sl].reshape(-1, sl.stop - sl.start).astype(np.complex128)
        if normalize:
            Ht = Ht / np.sqrt(np.mean(np.abs(Ht) ** 2, axis=1, keepdims=True))
            He = He / np.sqrt(np.mean(np.abs(He) ** 2, axis=1, keepdims=True))
        return Ht[fit_v], Ht[val_v], He

    def run_classic(Hf, Hv, He, p, snr, conv, seed):
        n = Hf.shape[1]
        miss = np.setdiff1d(np.arange(n), p)
        nv = 10 ** (-snr / 10) if conv == "per_link" else np.mean(np.abs(Hf[:, p]) ** 2) * 10 ** (-snr / 10)
        Ye = He[:, p] + np.sqrt(nv) * cn(np.random.default_rng(seed), (len(He), len(p)))
        Yv = Hv[:, p] + np.sqrt(nv) * cn(np.random.default_rng(seed + 1), (len(Hv), len(p)))
        mu, C = stats(Hf)
        rho = min(RHOS, key=lambda r: nmse(lmmse_apply(mu, lmmse_G(C, p, nv, r), Yv, p)[:, miss],
                                           Hv[:, miss]))
        est = {"linear": Ye @ linear_matrix(n, p).T,
               "lmmse": lmmse_apply(mu, lmmse_G(C, p, nv, rho), Ye, p)}
        if len(p) >= 4:
            est["spline"] = Ye @ spline_matrix(n, p).T
        mu_g, C_g = stats(He)
        est["genie"] = lmmse_apply(mu_g, lmmse_G(C_g, p, nv, rho), Ye, p)
        return est, miss, Ye, rho

    # ------------------------------------------------ 2. sweep
    print("\n[2] noisy pilots: NMSE (dB) on withheld subcarriers, eval run 0056")
    res["sweep"] = []
    for conv in ["per_link", "fixed_noise"]:
        for start in [128, 256, 384, 576, 704, 832]:
            sl = slice(start, start + 64)
            Hf, Hv, He = block(sl, conv == "per_link")
            for n_obs in [16, 8, 4]:
                p = pilot_positions(64, n_obs=n_obs)
                for snr in SNRS:
                    est, miss, _, rho = run_classic(Hf, Hv, He, p, snr, conv, 100 + snr)
                    r = {"conv": conv, "block": start, "n_obs": n_obs, "snr": snr, "rho": rho}
                    r.update({k: nmse(v[:, miss], He[:, miss]) for k, v in est.items()})
                    res["sweep"].append(r)
    for conv in ["per_link", "fixed_noise"]:
        print(f"  convention = {conv}: mean over 6 blocks [min..max of LMMSE gain over linear]")
        for n_obs in [16, 8, 4]:
            line = []
            for snr in [0, 10, 20, 30]:
                rs = [r for r in res["sweep"] if r["conv"] == conv and r["n_obs"] == n_obs and r["snr"] == snr]
                lin, lm, ge = (np.mean([r[k] for r in rs]) for k in ("linear", "lmmse", "genie"))
                gains = [r["linear"] - r["lmmse"] for r in rs]
                line.append(f"{snr:2d}dB lin {lin:6.1f} lmmse {lm:6.1f} genie {ge:6.1f} "
                            f"gain[{min(gains):4.1f}..{max(gains):4.1f}]")
            print(f"   {n_obs:2d}/64 | " + " | ".join(line))

    # ------------------------------------------------ main configuration (block 256, per-link)
    sl = slice(256, 320)
    Hf, Hv, He = block(sl, True)
    main = {}
    for n_obs in [16, 8]:
        p = pilot_positions(64, n_obs=n_obs)
        for snr in SNRS:
            est, miss, Ye, rho = run_classic(Hf, Hv, He, p, snr, "per_link", 100 + snr)
            main[(n_obs, snr)] = {"est": est, "miss": miss, "Ye": Ye, "p": p}

    # ------------------------------------------------ 7. MLP (needed before bootstrap)
    print("\n[7] numpy MLP 2x128 ReLU, trained over pilot SNR U[0,30] dB (per-link), block 256")
    seeds = [0, 1] if a.quick else [0, 1, 2]
    res["mlp"] = {}
    for n_obs in [16, 8]:
        p = main[(n_obs, 0)]["p"]
        miss = main[(n_obs, 0)]["miss"]
        runs = []
        for sd in seeds:
            net, vbest, secs = train_mlp(Hf, Hv, p, miss, sd)
            per_snr = {}
            for snr in SNRS:
                m = main[(n_obs, snr)]
                pred = mlp_predict(net, m["Ye"], snr, len(miss))
                per_snr[snr] = nmse(pred, He[:, miss])
                m["est"].setdefault("mlp_runs", []).append(pred)
            runs.append(per_snr)
            print(f"   {n_obs}/64 seed {sd}: val {vbest:6.2f} dB, train {secs:5.1f} s, params {net.n_params()}, "
                  + " ".join(f"{s}:{per_snr[s]:.1f}" for s in SNRS))
        res["mlp"][n_obs] = {"per_seed": runs, "params": net.n_params(),
                             "mean": {s: float(np.mean([r[s] for r in runs])) for s in SNRS},
                             "std": {s: float(np.std([r[s] for r in runs])) for s in SNRS}}
        t0 = time.time()
        for _ in range(20):
            mlp_predict(net, main[(n_obs, 10)]["Ye"], 10, len(miss))
        t_mlp = (time.time() - t0) / 20
        mu, C = stats(Hf)
        G = lmmse_G(C, p, 0.1, 1e-3)
        t0 = time.time()
        for _ in range(20):
            lmmse_apply(mu, G, main[(n_obs, 10)]["Ye"], p)
        t_lm = (time.time() - t0) / 20
        res["mlp"][n_obs]["time_per_vector_us"] = {"mlp": t_mlp / len(He) * 1e6, "lmmse": t_lm / len(He) * 1e6}

    print("\n   summary block 256-319, per-link, eval run 0056 (MLP = mean over seeds):")
    for n_obs in [16, 8]:
        for snr in SNRS:
            e = main[(n_obs, snr)]["est"]
            miss = main[(n_obs, snr)]["miss"]
            print(f"   {n_obs:2d}/64 {snr:2d} dB: linear {nmse(e['linear'][:, miss], He[:, miss]):6.1f}"
                  f"  spline {nmse(e['spline'][:, miss], He[:, miss]):6.1f}"
                  f"  LMMSE {nmse(e['lmmse'][:, miss], He[:, miss]):6.1f}"
                  f"  genie {nmse(e['genie'][:, miss], He[:, miss]):6.1f}"
                  f"  MLP {res['mlp'][n_obs]['mean'][snr]:6.1f} ± {res['mlp'][n_obs]['std'][snr]:.1f}")

    # ------------------------------------------------ 3. bootstrap CIs
    print("\n[3] bootstrap 95% CI over the 30 evaluation chunks (block 256, per-link)")
    rng = np.random.default_rng(7)
    B = 2000
    boots = rng.integers(0, n_eval_chunks, (B, n_eval_chunks))
    res["bootstrap"] = []
    for n_obs in [16, 8]:
        for snr in [0, 10, 20, 30]:
            m = main[(n_obs, snr)]
            miss = m["miss"]
            sums = {}
            for k in ["linear", "spline", "lmmse"]:
                sums[k] = sums_by(m["est"][k][:, miss], He[:, miss], ev_chunk, n_eval_chunks)
            # a single trained model (seed 0), not a seed ensemble
            sums["mlp_seed0"] = sums_by(m["est"]["mlp_runs"][0], He[:, miss], ev_chunk, n_eval_chunks)

            def boot(k):
                e, r = sums[k]
                return 10 * np.log10(e[boots].sum(1) / r[boots].sum(1))
            bl, bm, bn = boot("linear"), boot("lmmse"), boot("mlp_seed0")
            ci = lambda v: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
            r = {"n_obs": n_obs, "snr": snr, "linear": ci(bl), "lmmse": ci(bm), "mlp_seed0": ci(bn),
                 "gain_lmmse_over_linear": ci(bl - bm), "mlp_minus_lmmse": ci(bn - bm)}
            res["bootstrap"].append(r)
            print(f"   {n_obs:2d}/64 {snr:2d} dB: LMMSE gain over linear {r['gain_lmmse_over_linear']} dB,"
                  f" MLP(seed 0) minus LMMSE {r['mlp_minus_lmmse']} dB")

    # ------------------------------------------------ 4. antennas
    print("\n[4] spread over the 32 antennas (block 256, per-link, 10 dB)")
    res["antennas"] = {}
    for n_obs in [16, 8]:
        m = main[(n_obs, 10)]
        miss = m["miss"]
        per = {k: [] for k in ["linear", "lmmse"]}
        for ant in range(n_ant):
            sel = ev_ant == ant
            for k in per:
                per[k].append(nmse(m["est"][k][sel][:, miss], He[sel][:, miss]))
        gain = np.array(per["linear"]) - np.array(per["lmmse"])
        res["antennas"][n_obs] = {"lmmse_min_med_max": [round(float(f(per["lmmse"])), 2) for f in (np.min, np.median, np.max)],
                                  "gain_min_med_max": [round(float(f(gain)), 2) for f in (np.min, np.median, np.max)]}
        print(f"   {n_obs}/64: LMMSE NMSE min/med/max {res['antennas'][n_obs]['lmmse_min_med_max']},"
              f" gain over linear min/med/max {res['antennas'][n_obs]['gain_min_med_max']}")

    # ------------------------------------------------ 5. training size (LMMSE)
    print("\n[5] LMMSE versus number of fitting vectors (block 256, per-link, 10 dB, 5 random subsets)")
    res["train_size"] = []
    fit_chunks = np.unique(tr_chunk[fit_v])
    Hf_chunk = tr_chunk[fit_v]
    for n_obs in [16, 8]:
        p = pilot_positions(64, n_obs=n_obs)
        miss = np.setdiff1d(np.arange(64), p)
        Ye = main[(n_obs, 10)]["Ye"]
        for n_ch in [1, 2, 4, 8, 16, 40]:
            vals = []
            for rep in range(5 if n_ch < 40 else 1):
                pick = np.random.default_rng(rep).choice(fit_chunks, n_ch, replace=False)
                Hs = Hf[np.isin(Hf_chunk, pick)]
                mu, C = stats(Hs)
                rho = min(RHOS, key=lambda r: nmse(lmmse_apply(mu, lmmse_G(C, p, 0.1, r),
                                                               Hv[:, p] + np.sqrt(0.1) * cn(np.random.default_rng(9), (len(Hv), len(p))), p)[:, miss], Hv[:, miss]))
                vals.append(nmse(lmmse_apply(mu, lmmse_G(C, p, 0.1, rho), Ye, p)[:, miss], He[:, miss]))
            r = {"n_obs": n_obs, "chunks": n_ch, "vectors": int(n_ch * 3 * n_ant),
                 "nmse_mean": float(np.mean(vals)), "nmse_std": float(np.std(vals))}
            res["train_size"].append(r)
            print(f"   {n_obs}/64 {n_ch:2d} chunks ({r['vectors']:5d} vectors): {r['nmse_mean']:6.2f} ± {r['nmse_std']:.2f}")

    # MLP training-size effect (8/64)
    print("   MLP 8/64 versus training size (10 dB):")
    res["mlp_train_size"] = []
    p = pilot_positions(64, n_obs=8)
    miss = np.setdiff1d(np.arange(64), p)
    for frac_chunks in [10, 20, 40]:
        vals = []
        for sd in ([0] if a.quick else [0, 1]):
            pick = np.random.default_rng(sd).choice(fit_chunks, frac_chunks, replace=False)
            Hs = Hf[np.isin(Hf_chunk, pick)]
            net, _, _ = train_mlp(Hs, Hv, p, miss, sd, epochs=int(160 * 40 / frac_chunks))
            vals.append(nmse(mlp_predict(net, main[(8, 10)]["Ye"], 10, len(miss)), He[:, miss]))
        res["mlp_train_size"].append({"chunks": frac_chunks, "vectors": int(frac_chunks * 3 * n_ant),
                                      "nmse": float(np.mean(vals))})
        print(f"     {frac_chunks:2d} chunks ({frac_chunks * 3 * n_ant} vectors): {np.mean(vals):6.2f} dB")

    # ------------------------------------------------ 6. spatial shift
    print("\n[6] spatial shift: fit only on x > -2.5 m (run 0053), evaluate on both sides (run 0056), 8/64, 10 dB")
    p = pilot_positions(64, n_obs=8)
    miss = np.setdiff1d(np.arange(64), p)
    Ye = main[(8, 10)]["Ye"]
    xf = tr_x[fit_v]
    near_e = ev_x > -2.5
    res["spatial"] = {"n_fit_near": int(np.sum(xf > -2.5)), "n_fit_all": int(len(xf)),
                      "n_eval_near": int(near_e.sum()), "n_eval_far": int((~near_e).sum())}
    for name, Hs in [("fit_all", Hf), ("fit_near_only", Hf[xf > -2.5])]:
        mu, C = stats(Hs)
        est = lmmse_apply(mu, lmmse_G(C, p, 0.1, 1e-3), Ye, p)
        res["spatial"][name] = {"lmmse_near": nmse(est[near_e][:, miss], He[near_e][:, miss]),
                                "lmmse_far": nmse(est[~near_e][:, miss], He[~near_e][:, miss])}
    lin = Ye @ linear_matrix(64, p).T
    res["spatial"]["linear_near"] = nmse(lin[near_e][:, miss], He[near_e][:, miss])
    res["spatial"]["linear_far"] = nmse(lin[~near_e][:, miss], He[~near_e][:, miss])
    net_all, _, _ = train_mlp(Hf, Hv, p, miss, 0)
    net_near, _, _ = train_mlp(Hf[xf > -2.5], Hv, p, miss, 0, epochs=int(160 * len(Hf) / max(1, np.sum(xf > -2.5))))
    for name, net in [("mlp_fit_all", net_all), ("mlp_fit_near_only", net_near)]:
        pred = mlp_predict(net, Ye, 10, len(miss))
        res["spatial"][name] = {"near": nmse(pred[near_e], He[near_e][:, miss]),
                                "far": nmse(pred[~near_e], He[~near_e][:, miss])}
    print("   ", json.dumps(res["spatial"], indent=None))

    # in-run vs cross-run (does the cross-run test differ from held-out chunks of the same run?)
    p = pilot_positions(64, n_obs=8)
    miss = np.setdiff1d(np.arange(64), p)
    mu, C = stats(Hf)
    G = lmmse_G(C, p, 0.1, 1e-3)
    Yv = Hv[:, p] + np.sqrt(0.1) * cn(np.random.default_rng(3), (len(Hv), len(p)))
    res["in_vs_cross_run"] = {"heldout_chunks_0053": nmse(lmmse_apply(mu, G, Yv, p)[:, miss], Hv[:, miss]),
                              "run_0056": nmse(lmmse_apply(mu, G, main[(8, 10)]["Ye"], p)[:, miss], He[:, miss])}
    print("\n   LMMSE 8/64 10 dB: held-out chunks of 0053 vs run 0056:", res["in_vs_cross_run"])

    res["runtime_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.out, "rigor_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nsaved", os.path.join(a.out, "rigor_results.json"), f"({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
