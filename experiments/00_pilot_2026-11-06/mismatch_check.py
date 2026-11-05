"""Noise-level mismatch: what if the receiver does not know the pilot SNR exactly?

Screened data, fixed-noise convention, block 256-319.
  LMMSE (true SNR)       designed with the actual noise variance (reference)
  LMMSE (nominal X dB)   designed for a fixed SNR X, whatever the actual SNR
  MLP (SNR input)        trained over U[0,30] dB with the SNR as input
  MLP (blind)            trained over U[0,30] dB without any SNR input
  linear interpolation   needs no SNR

Usage:  python -I mismatch_check.py --cache DIR [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_variants import BatchLMMSE, train  # noqa: E402
from pilot_check import calibrate, linear_matrix, load_offsets, load_sample, pilot_positions  # noqa: E402
from rigor_check import cn, nmse  # noqa: E402
from screen_check import csi_error_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    tr = load_sample("0053", 50, 3, a.cache)
    ev = load_sample("0056", 30, 3, a.cache)
    sto, cpo = load_offsets(a.cache)
    tr["csi"], ev["csi"] = calibrate(tr["csi"], sto, cpo), calibrate(ev["csi"], sto, cpo)
    n_ant = tr["csi"].shape[1]
    fit_v = np.repeat((tr["chunk"] % 5) != 4, n_ant)
    keep_tr, keep_ev = csi_error_db(tr["csi"]) <= -30, csi_error_db(ev["csi"]) <= -30
    sl = slice(256, 320)
    Ht = tr["csi"][:, :, sl].reshape(-1, 64).astype(np.complex128)
    He = ev["csi"][:, :, sl].reshape(-1, 64).astype(np.complex128)[keep_ev]
    c = 1 / np.sqrt(np.mean(np.abs(Ht[fit_v & keep_tr]) ** 2))
    Ht, He = Ht * c, He * c
    Hf, Hv = Ht[fit_v & keep_tr], Ht[~fit_v & keep_tr]

    res = {}
    for n_obs in [16, 8]:
        p = pilot_positions(64, n_obs=n_obs)
        miss = np.setdiff1d(np.arange(64), p)
        lm = BatchLMMSE(Hf, p)
        mlp_snr = train(Hf, Hv, p, miss, 0)
        mlp_blind = train(Hf, Hv, p, miss, 0, blind=True)
        rows = {}
        for snr in [0, 5, 10, 15, 20, 25, 30]:
            nv = 10 ** (-snr / 10)
            Ye = He[:, p] + np.sqrt(nv) * cn(np.random.default_rng(500 + snr), (len(He), len(p)))
            s = np.full(len(He), float(snr))
            r = {"linear": nmse((Ye @ linear_matrix(64, p).T)[:, miss], He[:, miss]),
                 "lmmse_true": nmse(lm(Ye, nv)[:, miss], He[:, miss]),
                 "lmmse_nom10": nmse(lm(Ye, 10 ** (-1.0))[:, miss], He[:, miss]),
                 "lmmse_nom20": nmse(lm(Ye, 10 ** (-2.0))[:, miss], He[:, miss]),
                 "lmmse_nom30": nmse(lm(Ye, 10 ** (-3.0))[:, miss], He[:, miss]),
                 "mlp_snr_input": nmse(mlp_snr(Ye, s), He[:, miss]),
                 "mlp_blind": nmse(mlp_blind(Ye, s), He[:, miss])}
            rows[snr] = {k: round(v, 2) for k, v in r.items()}
            print(f"{n_obs:2d}/64 {snr:2d} dB: " + "  ".join(f"{k} {v:6.1f}" for k, v in r.items()))
        res[str(n_obs)] = rows
    with open(os.path.join(a.out, "mismatch_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.join(a.out, "mismatch_results.json"))


if __name__ == "__main__":
    main()
