"""Companion to gmm_feasibility.py (setting spacefreq, pilots on all 32 antennas x 4 of 16 subcarriers).

How much of the joint result comes from processing the antennas jointly at all?
  linear-per-antenna   linear interpolation along frequency, each antenna separately
  lmmse-per-antenna    LMMSE with each antenna's own 16x16 frequency covariance (block-diagonal),
                       i.e. the main-line estimator applied antenna by antenna
  lmmse-joint          LMMSE with the full 512x512 covariance (same as "lmmse" in results_spacefreq.json)
Same data split, scale, noise seeds and selection protocol as gmm_feasibility.py.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from estimators import add_noise, linear_matrix, pilot_positions  # noqa: E402
from gmm_feasibility import RHOS, SNRS, cov, load, mix_estimate  # noqa: E402


def main():
    D = load(os.path.join(ROOT, "data", "processed", "dichasus005x_sc256-319.npz"), "spacefreq")
    Xf, Xv, sel, rep = D["Xf"], D["Xv"], D["sel"], D["rep"]
    d = Xf.shape[1]
    Cg = cov(Xf)
    Cg = Cg + 1e-6 * np.real(np.trace(Cg)) / d * np.eye(d)
    Cbd = Cg * np.kron(np.eye(32), np.ones((16, 16)))
    pname = "32ant x 4sc"
    P = D["pats"][pname]
    M = np.setdiff1d(np.arange(d), P)
    sc_p = pilot_positions(16, 4)
    W = linear_matrix(16, sc_p)                                   # (16, 4)
    nm = lambda E, H: float(10 * np.log10(np.sum(np.abs(E[:, M] - H[:, M]) ** 2) / np.sum(np.abs(H[:, M]) ** 2)))
    out = {}
    for snr in SNRS:
        nv = 10 ** (-snr / 10)
        Y = add_noise(Xv[:, P], nv, seed=7919 * len(P) + snr)     # identical to gmm_feasibility.py
        row = {}
        lin = (Y.reshape(len(Y), 32, 4) @ W.T).reshape(len(Y), -1)
        row["linear-per-antenna"] = round(nm(lin[rep], Xv[rep]), 3)
        for name, C in (("lmmse-per-antenna", Cbd), ("lmmse-joint", Cg)):
            rho = min(RHOS, key=lambda r: nm(mix_estimate(Y[sel], P, np.ones(1), C[None], nv, r), Xv[sel]))
            row[name] = round(nm(mix_estimate(Y[rep], P, np.ones(1), C[None], nv, rho), Xv[rep]), 3)
        out[str(snr)] = row
        print(f"{snr:2d} dB: " + "  ".join(f"{k} {v:7.2f}" for k, v in row.items()), flush=True)
    with open(os.path.join(HERE, "results_ladder_spacefreq.json"), "w", encoding="utf-8") as f:
        json.dump({"pattern": pname, "nmse_db_report_blocks": out,
                   "note": "validation run 0056, odd blocks; per-antenna = block-diagonal covariance"}, f, indent=1)


if __name__ == "__main__":
    main()
