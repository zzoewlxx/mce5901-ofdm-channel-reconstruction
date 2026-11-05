"""Feasibility check for 'OFDM Channel Reconstruction from Sparse Measured CSI'.

Uses a small sample of dichasus-005x fetched with HTTP Range requests:
  * run 0053 (a training run in the proposal): evenly spaced chunks -> fit
  * run 0056 (the validation run in the proposal): evenly spaced chunks -> evaluate
Run 0057 (the proposal's test run) is deliberately NOT touched.

All 32 antennas are pooled as samples here to stand in for the ~16k training
snapshots the full study will have. Results are a pilot, not final numbers.

Usage:  python -I pilot_check.py --cache DIR [--out DIR]
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from dichasus_reader import fetch_records, resolve_run  # noqa: E402

OFFSETS_URL = ("https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-005x/"
               "reftx-offsets-dichasus-005x.json")
NSC = 1024
DF = 50e6 / NSC  # subcarrier spacing, Hz


# ------------------------------------------------------------------ data
def load_sample(run, n_chunks, chunk_len, cache):
    fn = os.path.join(cache, f"dichasus-{run}_{n_chunks}x{chunk_len}.npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=False)
        return {k: z[k] for k in z.files}
    _, _, _, n_rec = resolve_run(run)
    starts = np.linspace(0, n_rec - chunk_len, n_chunks).astype(int)
    recs, _ = fetch_records(run, starts, chunk_len)
    d = {"csi": np.stack([r["csi"] for r in recs]),
         "snr": np.stack([r["snr"] for r in recs]),
         "pos": np.stack([r["pos"] for r in recs]),
         "time": np.array([r["time"] for r in recs]),
         "index": np.array([r["index"] for r in recs]),
         "chunk": np.repeat(np.arange(n_chunks), chunk_len)[:len(recs)]}
    np.savez(fn, **d)
    return d


def load_offsets(cache):
    fn = os.path.join(cache, "reftx-offsets-dichasus-005x.json")
    if not os.path.exists(fn):
        with urllib.request.urlopen(OFFSETS_URL, timeout=60) as r, open(fn, "wb") as f:
            f.write(r.read())
    with open(fn, encoding="utf-8") as f:
        off = json.load(f)
    return np.asarray(off["sto"]), np.asarray(off["cpo"])


def calibrate(csi, sto, cpo):
    """Same correction as the official REFTXOffsetCalibration tutorial (after fftshift)."""
    k = np.arange(csi.shape[-1])
    phase = sto[:, None] * 2 * np.pi * k[None, :] / csi.shape[-1] + cpo[:, None]
    return (csi * np.exp(1j * phase)[None]).astype(np.complex64)


# ------------------------------------------------------------- estimators
def nmse_db(est, ref):
    return float(10 * np.log10(np.sum(np.abs(est - ref) ** 2) / np.sum(np.abs(ref) ** 2)))


def pilot_positions(n, spacing=None, n_obs=None):
    if n_obs is not None:  # approximately uniform incl. both endpoints (proposal)
        return np.unique(np.round(np.linspace(0, n - 1, n_obs)).astype(int))
    p = np.arange(0, n, spacing)
    return p if p[-1] == n - 1 else np.append(p, n - 1)


def linear_matrix(n, p):
    eye = np.eye(len(p))
    return np.stack([np.interp(np.arange(n), p, eye[j]) for j in range(len(p))], axis=1)


def spline_matrix(n, p):
    """Natural cubic spline interpolation as an (n x len(p)) matrix (no scipy needed)."""
    x = p.astype(float)
    m, h = len(x), np.diff(x)
    xs = np.arange(n, dtype=float)
    seg = np.clip(np.searchsorted(x, xs, side="right") - 1, 0, m - 2)
    W = np.zeros((n, m))
    for j in range(m):
        y = np.zeros(m)
        y[j] = 1.0
        A = np.zeros((m, m))
        rhs = np.zeros(m)
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, m - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        M = np.linalg.solve(A, rhs)
        i = seg
        t0, t1, hi = xs - x[i], x[i + 1] - xs, h[i]
        W[:, j] = (M[i] * t1 ** 3 + M[i + 1] * t0 ** 3) / (6 * hi) \
            + (y[i] / hi - M[i] * hi / 6) * t1 + (y[i + 1] / hi - M[i + 1] * hi / 6) * t0
    return W


class LMMSE:
    """Data-driven LMMSE interpolation = linear regression from pilots to all subcarriers.

    mean / covariance are estimated from training vectors only; diagonal loading
    `rho` (relative to mean pilot power) is chosen on a validation split.
    """

    def __init__(self, H_train, p):
        self.p = p
        self.mu = H_train.mean(0)
        X = H_train - self.mu                    # rows are samples
        self.C = (X.T @ X.conj()) / len(X)       # C[j,k] = E[(h_j-mu_j)(h_k-mu_k)^*]
        self.ppow = float(np.real(np.trace(self.C[np.ix_(p, p)])) / len(p))

    def matrix(self, noise_var=0.0, rho=1e-3):
        p = self.p
        Cpp = self.C[np.ix_(p, p)] + (noise_var + rho * self.ppow) * np.eye(len(p))
        return np.linalg.solve(Cpp.T, self.C[:, p].T).T  # C[:,p] @ inv(Cpp)

    def predict(self, Y_p, G):
        return self.mu + (Y_p - self.mu[self.p]) @ G.T


def evaluate(H_fit, H_val, H_eval, p, n, noise_var=0.0, rng=None, rhos=None):
    """Return NMSE(dB) on withheld subcarriers for linear / spline / LMMSE."""
    miss = np.setdiff1d(np.arange(n), p)
    rng = rng or np.random.default_rng(0)

    def noisy(H):
        Y = H[:, p]
        if noise_var > 0:
            Y = Y + np.sqrt(noise_var / 2) * (rng.standard_normal(Y.shape)
                                             + 1j * rng.standard_normal(Y.shape))
        return Y

    lm = LMMSE(H_fit, p)
    rhos = rhos if rhos is not None else [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
    Yv = noisy(H_val)
    best = min(rhos, key=lambda r: nmse_db(lm.predict(Yv, lm.matrix(noise_var, r))[:, miss],
                                           H_val[:, miss]))
    Ye = noisy(H_eval)
    out = {"linear": nmse_db((Ye @ linear_matrix(n, p).T)[:, miss], H_eval[:, miss]),
           "lmmse": nmse_db(lm.predict(Ye, lm.matrix(noise_var, best))[:, miss], H_eval[:, miss]),
           "lmmse_rho": best}
    if len(p) >= 4:
        out["spline"] = nmse_db((Ye @ spline_matrix(n, p).T)[:, miss], H_eval[:, miss])
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="directory for the downloaded sample")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--train-chunks", type=int, default=50)
    ap.add_argument("--eval-chunks", type=int, default=30)
    ap.add_argument("--chunk-len", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    res = {}

    tr = load_sample("0053", a.train_chunks, a.chunk_len, a.cache)
    ev = load_sample("0056", a.eval_chunks, a.chunk_len, a.cache)
    sto, cpo = load_offsets(a.cache)
    tr["csi"], ev["csi"] = calibrate(tr["csi"], sto, cpo), calibrate(ev["csi"], sto, cpo)
    print(f"sample: 0053 {tr['csi'].shape}, 0056 {ev['csi'].shape}")
    res["sample"] = {"train_snapshots": int(len(tr["csi"])), "eval_snapshots": int(len(ev["csi"]))}

    # ---- A. measured SNR field, CSI noise floor, delay profile -----------------
    allc = np.concatenate([tr["csi"], ev["csi"]]).reshape(-1, NSC)
    snr_field = np.concatenate([tr["snr"], ev["snr"]]).ravel()
    g = np.fft.ifft(np.fft.ifftshift(allc, axes=-1), axis=-1, norm="ortho")
    noise_var = np.mean(np.abs(g[:, 256:768]) ** 2, axis=1)       # 5-15 us: no multipath
    band = np.r_[64:512, 513:960]
    sig = np.mean(np.abs(allc[:, band]) ** 2, axis=1)
    csi_snr = 10 * np.log10(sig / noise_var)
    pdp = np.mean(np.abs(g) ** 2 / sig[:, None], axis=0)
    pct = lambda v: [round(float(x), 1) for x in np.percentile(v, [10, 25, 50, 75, 90])]
    res["snr_field_pct_10_25_50_75_90"] = pct(snr_field)
    res["csi_snr_from_delay_domain_pct"] = pct(csi_snr)
    res["corr_snrfield_csisnr"] = float(np.corrcoef(snr_field, csi_snr)[0, 1])
    print("snr field  (dB) p10/25/50/75/90:", res["snr_field_pct_10_25_50_75_90"])
    print("CSI SNR from delay-domain noise (dB):", res["csi_snr_from_delay_domain_pct"],
          " corr with snr field: %.2f" % res["corr_snrfield_csisnr"])

    # magnitude profile over subcarriers (band edges, DC)
    mag_db = 10 * np.log10(np.mean(np.abs(allc) ** 2, axis=0))
    mag_db -= np.median(mag_db[band])
    res["mean_power_profile_db"] = {int(k): round(float(mag_db[k]), 2)
                                    for k in [0, 16, 32, 64, 128, 256, 500, 510, 511, 512, 513,
                                              514, 768, 896, 960, 992, 1008, 1023]}
    print("mean |H|^2 profile vs band median (dB):", res["mean_power_profile_db"])

    # frequency correlation over the clean band -> coherence bandwidth
    Hc = allc[:, 64:512]
    lags = np.arange(0, 400)
    fcf = np.array([np.abs(np.sum(Hc[:, :Hc.shape[1] - L] * np.conj(Hc[:, L:])))
                    / np.sqrt(np.sum(np.abs(Hc[:, :Hc.shape[1] - L]) ** 2)
                              * np.sum(np.abs(Hc[:, L:]) ** 2)) for L in lags])
    b90 = int(lags[np.argmax(fcf < 0.9)]) if np.any(fcf < 0.9) else None
    b50 = int(lags[np.argmax(fcf < 0.5)]) if np.any(fcf < 0.5) else None
    res["fcf"] = {"lags": lags.tolist(), "abs_corr": np.round(fcf, 4).tolist()}
    res["coherence_bw"] = {"lag90_subc": b90, "lag50_subc": b50,
                           "Bc90_MHz": None if b90 is None else round(b90 * DF / 1e6, 2),
                           "Bc50_MHz": None if b50 is None else round(b50 * DF / 1e6, 2)}
    print("frequency correlation |R|: lag 8 -> %.3f, lag 16 -> %.3f, lag 32 -> %.3f, lag 64 -> %.3f"
          % (fcf[8], fcf[16], fcf[32], fcf[64]))
    print("coherence bandwidth:", res["coherence_bw"])
    res["pdp_first_64_taps_db"] = np.round(10 * np.log10(pdp[:64] + 1e-12), 2).tolist()
    res["pdp_noise_level_db"] = round(float(10 * np.log10(np.mean(pdp[256:768]))), 2)

    # ---- B/C/D. reconstruction experiments -----------------------------------
    fit_mask = (tr["chunk"] % 5) != 4           # interleaved chunk split for rho selection
    Htr_all, Hev_all = tr["csi"], ev["csi"]
    rng = np.random.default_rng(1)

    def vectors(H, sl, mask=None):
        H = H if mask is None else H[mask]
        return H[:, :, sl].reshape(-1, sl.stop - sl.start)

    # B. proposal configuration: 64-subcarrier blocks, 8 or 16 observed (endpoints included)
    res["proposal_cfg"] = []
    print("\n[B] proposal config: 64-subcarrier block, noiseless pilots (NMSE dB on withheld)")
    for start in [128, 256, 384, 576, 704, 832]:
        sl = slice(start, start + 64)
        Hf, Hv = vectors(Htr_all, sl, fit_mask), vectors(Htr_all, sl, ~fit_mask)
        He = vectors(Hev_all, sl)
        g_e = np.fft.ifft(np.fft.ifftshift(Hev_all, axes=-1), axis=-1, norm="ortho")
        floor = float(10 * np.log10(np.mean(np.abs(g_e[..., 256:768]) ** 2)
                                    / np.mean(np.abs(He) ** 2)))
        for n_obs in (16, 8, 4):
            p = pilot_positions(64, n_obs=n_obs)
            r = evaluate(Hf, Hv, He, p, 64, rng=rng)
            r.update(block_start=start, n_obs=n_obs, floor=round(floor, 2))
            res["proposal_cfg"].append(r)
            print(f"  block {start:4d}-{start + 63:4d}  obs {n_obs:2d}/64: linear {r['linear']:6.1f}"
                  f"  spline {r.get('spline', float('nan')):6.1f}  LMMSE {r['lmmse']:6.1f}"
                  f"   (noise floor ~{floor:6.1f})")

    # C. wider block, sparser pilots
    print("\n[C] wide block 64-511 (448 subcarriers = 21.9 MHz), pilot spacing sweep, noiseless pilots")
    sl = slice(64, 512)
    Hf, Hv, He = vectors(Htr_all, sl, fit_mask), vectors(Htr_all, sl, ~fit_mask), vectors(Hev_all, sl)
    res["wide_cfg"] = []
    for D in (4, 8, 16, 32, 48, 64, 96):
        p = pilot_positions(448, spacing=D)
        r = evaluate(Hf, Hv, He, p, 448, rng=rng)
        r.update(spacing=D, n_obs=int(len(p)), spacing_kHz=round(D * DF / 1e3, 1))
        res["wide_cfg"].append(r)
        print(f"  spacing {D:3d} ({D * DF / 1e3:6.0f} kHz, {len(p):3d} pilots): linear {r['linear']:6.1f}"
              f"  spline {r.get('spline', float('nan')):6.1f}  LMMSE {r['lmmse']:6.1f}")

    # D. noisy pilots (emulates LS estimates at pilot positions)
    print("\n[D] noisy pilots: pilot SNR relative to average channel power")
    res["noisy_cfg"] = []
    cfgs = [("block64_obs16", slice(256, 320), dict(n_obs=16)),
            ("block64_obs8", slice(256, 320), dict(n_obs=8)),
            ("wide448_D16", slice(64, 512), dict(spacing=16)),
            ("wide448_D32", slice(64, 512), dict(spacing=32))]
    for name, sl, kw in cfgs:
        n = sl.stop - sl.start
        Hf, Hv, He = vectors(Htr_all, sl, fit_mask), vectors(Htr_all, sl, ~fit_mask), vectors(Hev_all, sl)
        p = pilot_positions(n, **kw)
        pw = float(np.mean(np.abs(Hf[:, p]) ** 2))
        for snr_db in (0, 5, 10, 15, 20, 30):
            r = evaluate(Hf, Hv, He, p, n, noise_var=pw / 10 ** (snr_db / 10), rng=rng)
            r.update(cfg=name, pilot_snr_db=snr_db)
            res["noisy_cfg"].append(r)
            print(f"  {name:14s} SNR {snr_db:2d} dB: linear {r['linear']:6.1f}"
                  f"  spline {r.get('spline', float('nan')):6.1f}  LMMSE {r['lmmse']:6.1f}")

    # natural SNR groups (proposal: summarize errors by measured SNR), linear interp 8/64
    sl, p = slice(256, 320), pilot_positions(64, n_obs=8)
    miss = np.setdiff1d(np.arange(64), p)
    He = vectors(Hev_all, sl)
    est = He[:, p] @ linear_matrix(64, p).T
    s = ev["snr"].ravel()
    res["by_snr_field_linear_obs8"] = {}
    for lo, hi in [(-99, 10), (10, 15), (15, 20), (20, 25), (25, 99)]:
        m = (s >= lo) & (s < hi)
        if m.sum() > 20:
            res["by_snr_field_linear_obs8"][f"{lo}..{hi}"] = {
                "n": int(m.sum()), "nmse_db": round(nmse_db(est[m][:, miss], He[m][:, miss]), 2)}
    print("\nlinear 8/64 by measured SNR field:", res["by_snr_field_linear_obs8"])

    with open(os.path.join(a.out, "pilot_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nsaved", os.path.join(a.out, "pilot_results.json"))


if __name__ == "__main__":
    main()
