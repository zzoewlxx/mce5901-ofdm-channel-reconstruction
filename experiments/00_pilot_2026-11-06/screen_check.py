"""Follow-up to rigor_check.py: are a few noise-dominated snapshots distorting the results?

Per (snapshot, antenna) vector, the error level of the measured CSI is estimated from
4th-order differences over the clean band (128-893, DC excluded). Vectors whose
estimated error is above a threshold are screened out of fit / val / eval, and the
main comparisons (block 256-319, per-link SNR, 16/64 and 8/64) are repeated.

Usage:  python -I screen_check.py --cache DIR [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_check import calibrate, linear_matrix, load_offsets, load_sample, pilot_positions  # noqa: E402
from rigor_check import RHOS, cn, lmmse_apply, lmmse_G, mlp_predict, nmse, stats, train_mlp  # noqa: E402


def csi_error_db(csi):
    """(..., 1024) calibrated CSI -> per-vector error level relative to in-band power (dB)."""
    H = csi.reshape(-1, 1024).astype(np.complex128)
    d4 = []
    for lo, hi in [(130, 508), (517, 892)]:
        k = np.arange(lo, hi)
        d4.append(H[:, k - 2] - 4 * H[:, k - 1] + 6 * H[:, k] - 4 * H[:, k + 1] + H[:, k + 2])
    s = np.mean(np.abs(np.concatenate(d4, 1)) ** 2, axis=1) / 70.0
    band = np.r_[128:510, 515:894]
    return 10 * np.log10(s / np.mean(np.abs(H[:, band]) ** 2, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    res = {}

    tr = load_sample("0053", 50, 3, a.cache)
    ev = load_sample("0056", 30, 3, a.cache)
    sto, cpo = load_offsets(a.cache)
    tr["csi"], ev["csi"] = calibrate(tr["csi"], sto, cpo), calibrate(ev["csi"], sto, cpo)
    n_ant = tr["csi"].shape[1]
    fit_v = np.repeat((tr["chunk"] % 5) != 4, n_ant)
    val_v = ~fit_v

    q_tr, q_ev = csi_error_db(tr["csi"]), csi_error_db(ev["csi"])
    snr_tr, snr_ev = tr["snr"].ravel(), ev["snr"].ravel()
    q = np.concatenate([q_tr, q_ev])
    s = np.concatenate([snr_tr, snr_ev])
    pct = [1, 5, 10, 25, 50, 75, 90]
    res["error_db_percentiles"] = dict(zip(map(str, pct), np.round(np.percentile(q, pct), 1).tolist()))
    res["fraction_above"] = {str(t): round(float(np.mean(q > t)), 4) for t in (-40, -35, -30, -25, -20, -10)}
    res["corr_error_vs_snrfield"] = round(float(np.corrcoef(q, s)[0, 1]), 3)
    res["snrfield_of_flagged_-25"] = np.round(np.percentile(s[q > -25], [10, 50, 90]), 1).tolist() if np.any(q > -25) else None
    res["snrfield_of_clean"] = np.round(np.percentile(s[q <= -25], [10, 50, 90]), 1).tolist()
    print("CSI error level per vector (dB) percentiles:", res["error_db_percentiles"])
    print("fraction of vectors with error above threshold:", res["fraction_above"])
    print("corr(error level, snr field) =", res["corr_error_vs_snrfield"],
          "| snr field of flagged (>-25 dB) p10/50/90:", res["snrfield_of_flagged_-25"],
          "| of clean:", res["snrfield_of_clean"])
    # which snapshots are bad: whole snapshots or single antennas?
    bad_tr = (q_tr > -25).reshape(-1, n_ant)
    bad_ev = (q_ev > -25).reshape(-1, n_ant)
    res["bad_antennas_per_snapshot_hist"] = np.bincount(np.concatenate([bad_tr.sum(1), bad_ev.sum(1)]),
                                                        minlength=n_ant + 1).tolist()
    print("number of flagged antennas per snapshot (histogram 0..32):", res["bad_antennas_per_snapshot_hist"])

    sl = slice(256, 320)

    def vecs(csi):
        H = csi[:, :, sl].reshape(-1, 64).astype(np.complex128)
        return H / np.sqrt(np.mean(np.abs(H) ** 2, axis=1, keepdims=True))

    Ht, He_all = vecs(tr["csi"]), vecs(ev["csi"])
    res["screened"] = {}
    for thr in [None, -30.0, -25.0]:
        keep_tr = np.ones(len(Ht), bool) if thr is None else q_tr <= thr
        keep_ev = np.ones(len(He_all), bool) if thr is None else q_ev <= thr
        Hf, Hv, He = Ht[fit_v & keep_tr], Ht[val_v & keep_tr], He_all[keep_ev]
        key = "none" if thr is None else f"{thr:.0f}dB"
        res["screened"][key] = {"n_fit": int(len(Hf)), "n_val": int(len(Hv)), "n_eval": int(len(He))}
        print(f"\nscreen {key}: fit {len(Hf)}, val {len(Hv)}, eval {len(He)} vectors")
        for n_obs in [16, 8]:
            p = pilot_positions(64, n_obs=n_obs)
            miss = np.setdiff1d(np.arange(64), p)
            mu, C = stats(Hf)
            mu_g, C_g = stats(He)
            net, _, _ = train_mlp(Hf, Hv, p, miss, 0)
            rows = {}
            for snr in [0, 10, 20, 25, 30]:
                nv = 10 ** (-snr / 10)
                Ye = He[:, p] + np.sqrt(nv) * cn(np.random.default_rng(100 + snr), (len(He), len(p)))
                Yv = Hv[:, p] + np.sqrt(nv) * cn(np.random.default_rng(101 + snr), (len(Hv), len(p)))
                rho = min(RHOS, key=lambda r: nmse(lmmse_apply(mu, lmmse_G(C, p, nv, r), Yv, p)[:, miss], Hv[:, miss]))
                r = {"linear": nmse((Ye @ linear_matrix(64, p).T)[:, miss], He[:, miss]),
                     "lmmse": nmse(lmmse_apply(mu, lmmse_G(C, p, nv, rho), Ye, p)[:, miss], He[:, miss]),
                     "genie": nmse(lmmse_apply(mu_g, lmmse_G(C_g, p, nv, rho), Ye, p)[:, miss], He[:, miss]),
                     "mlp": nmse(mlp_predict(net, Ye, snr, len(miss)), He[:, miss])}
                rows[snr] = {k: round(v, 2) for k, v in r.items()}
                print(f"   {n_obs:2d}/64 {snr:2d} dB: linear {r['linear']:6.1f}  LMMSE {r['lmmse']:6.1f}"
                      f"  genie {r['genie']:6.1f}  MLP {r['mlp']:6.1f}")
            # LMMSE vs amount of fitting data at 10 dB
            curve = []
            for n_vec in [96, 384, 1536, len(Hf)]:
                vals = []
                for rep in range(5 if n_vec < len(Hf) else 1):
                    idx = np.random.default_rng(rep).choice(len(Hf), n_vec, replace=False)
                    mu_s, C_s = stats(Hf[idx])
                    Ye = He[:, p] + np.sqrt(0.1) * cn(np.random.default_rng(110), (len(He), len(p)))
                    vals.append(nmse(lmmse_apply(mu_s, lmmse_G(C_s, p, 0.1, 1e-3), Ye, p)[:, miss], He[:, miss]))
                curve.append([n_vec, round(float(np.mean(vals)), 2)])
            print(f"   {n_obs:2d}/64 LMMSE at 10 dB vs fitting vectors:", curve)
            res["screened"][key][str(n_obs)] = {"by_snr": rows, "lmmse_vs_fit_vectors_10dB": curve}

    with open(os.path.join(a.out, "screen_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nsaved", os.path.join(a.out, "screen_results.json"))


if __name__ == "__main__":
    main()
