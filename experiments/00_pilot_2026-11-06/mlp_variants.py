"""Last check before writing the proposal: is the MLP gain real, and can its high-SNR floor be fixed?

* Screened data (per-vector CSI error <= -30 dB), block 256-319.
* Two SNR conventions:
    per_link    each vector scaled to unit power with its TRUE power (the estimator
                implicitly knows the channel norm - possible leakage)
    fixed_noise one global scale from the training set; noise power fixed, so the
                per-snapshot SNR varies with the channel gain (no norm leakage)
* Estimators: linear, LMMSE (training statistics), MLP (trained over SNR U[0,30] dB with the
  SNR as input), residual MLP (LMMSE estimate + learned correction), SNR-specific MLP.

Usage:  python -I mlp_variants.py --cache DIR [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_check import calibrate, linear_matrix, load_offsets, load_sample, pilot_positions  # noqa: E402
from rigor_check import MLP, cn, mlp_features, nmse, stats  # noqa: E402
from screen_check import csi_error_db  # noqa: E402

SNRS = [0, 10, 20, 30]


class BatchLMMSE:
    """LMMSE with a per-row noise variance (eigen-decomposition of C_pp)."""

    def __init__(self, H, p, rho=1e-3):
        self.mu, C = stats(H)
        self.p = p
        lam, self.U = np.linalg.eigh(C[np.ix_(p, p)])
        self.lam = lam + rho * np.real(np.trace(C[np.ix_(p, p)])) / len(p)
        self.A = C[:, p] @ self.U

    def __call__(self, Y, nv):
        z = (Y - self.mu[self.p]) @ self.U.conj()
        z = z / (self.lam[None, :] + np.broadcast_to(nv, (len(Y),))[:, None])
        return self.mu + z @ self.A.T


def train(Hf, Hv, p, miss, seed, base=None, snr_fixed=None, epochs=160, batch=128, blind=False):
    """Train an MLP; with `base`, the MLP learns the residual h - base(y).

    blind=True hides the SNR from the network (its SNR input is always 0).
    """
    rng = np.random.default_rng(seed)
    net = MLP(2 * len(p) + 1, 2 * len(miss), 128, rng)
    val_snrs = SNRS if snr_fixed is None else [snr_fixed]
    Zv = {s: cn(np.random.default_rng(7000 + s), (len(Hv), len(p))) for s in val_snrs}

    def feats(Y, s):
        return mlp_features(Y, np.zeros(len(Y)) if blind else s)

    def predict(Y, s):
        out, _ = net.forward(feats(Y, s))
        r = out[:, :len(miss)] + 1j * out[:, len(miss):]
        return r if base is None else base(Y, 10 ** (-s / 10))[:, miss] + r

    best, best_P = np.inf, None
    for ep in range(epochs):
        lr = 1e-3 if ep < 0.6 * epochs else (3e-4 if ep < 0.85 * epochs else 1e-4)
        idx = rng.permutation(len(Hf))
        for b in range(0, len(idx), batch):
            h = Hf[idx[b:b + batch]]
            s = rng.uniform(0, 30, len(h)) if snr_fixed is None else np.full(len(h), float(snr_fixed))
            nv = 10 ** (-s / 10)
            Y = h[:, p] + np.sqrt(nv)[:, None] * cn(rng, (len(h), len(p)))
            target = h[:, miss] if base is None else h[:, miss] - base(Y, nv)[:, miss]
            out, cache = net.forward(feats(Y, s))
            T = np.concatenate([target.real, target.imag], axis=1)
            net.step(cache, 2 * (out - T) / out.size, lr)
        if ep % 4 == 3 or ep == epochs - 1:
            v = np.mean([nmse(predict(Hv[:, p] + np.sqrt(10 ** (-s / 10)) * Zv[s], np.full(len(Hv), s)),
                              Hv[:, miss]) for s in val_snrs])
            if v < best:
                best, best_P = v, [w.copy() for w in net.P]
    net.P = best_P
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    t0 = time.time()

    tr = load_sample("0053", 50, 3, a.cache)
    ev = load_sample("0056", 30, 3, a.cache)
    sto, cpo = load_offsets(a.cache)
    tr["csi"], ev["csi"] = calibrate(tr["csi"], sto, cpo), calibrate(ev["csi"], sto, cpo)
    n_ant = tr["csi"].shape[1]
    fit_v = np.repeat((tr["chunk"] % 5) != 4, n_ant)
    keep_tr, keep_ev = csi_error_db(tr["csi"]) <= -30, csi_error_db(ev["csi"]) <= -30
    sl = slice(256, 320)
    Ht = tr["csi"][:, :, sl].reshape(-1, 64).astype(np.complex128)
    He0 = ev["csi"][:, :, sl].reshape(-1, 64).astype(np.complex128)[keep_ev]

    res = {}
    for conv in ["fixed_noise", "per_link"]:
        if conv == "per_link":
            Hn = Ht / np.sqrt(np.mean(np.abs(Ht) ** 2, axis=1, keepdims=True))
            He = He0 / np.sqrt(np.mean(np.abs(He0) ** 2, axis=1, keepdims=True))
        else:
            c = 1 / np.sqrt(np.mean(np.abs(Ht[fit_v & keep_tr]) ** 2))
            Hn, He = Ht * c, He0 * c
        Hf, Hv = Hn[fit_v & keep_tr], Hn[~fit_v & keep_tr]
        res[conv] = {}
        for n_obs in [16, 8]:
            p = pilot_positions(64, n_obs=n_obs)
            miss = np.setdiff1d(np.arange(64), p)
            lm = BatchLMMSE(Hf, p)
            models = {"mlp": train(Hf, Hv, p, miss, 0),
                      "mlp_residual": train(Hf, Hv, p, miss, 0, base=lm)}
            spec = {s: train(Hf, Hv, p, miss, 0, snr_fixed=s) for s in (20, 30)}
            rows = {}
            for snr in SNRS:
                nv = 10 ** (-snr / 10)
                Ye = He[:, p] + np.sqrt(nv) * cn(np.random.default_rng(300 + snr), (len(He), len(p)))
                svec = np.full(len(He), float(snr))
                r = {"linear": nmse((Ye @ linear_matrix(64, p).T)[:, miss], He[:, miss]),
                     "lmmse": nmse(lm(Ye, nv)[:, miss], He[:, miss]),
                     "mlp": nmse(models["mlp"](Ye, svec), He[:, miss]),
                     "mlp_residual": nmse(models["mlp_residual"](Ye, svec), He[:, miss])}
                if snr in spec:
                    r["mlp_snr_specific"] = nmse(spec[snr](Ye, svec), He[:, miss])
                rows[snr] = {k: round(v, 2) for k, v in r.items()}
                print(f"{conv:11s} {n_obs:2d}/64 {snr:2d} dB: " +
                      "  ".join(f"{k} {v:6.1f}" for k, v in r.items()))
            res[conv][str(n_obs)] = rows
    res["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "mlp_variants_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.join(a.out, "mlp_variants_results.json"), f"({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
