"""Full-data baselines: linear interpolation, natural cubic spline, LMMSE (training statistics).

Protocol (matches the proposal):
  * input: data/processed/dichasus005x_sc256-319.npz from tools/extract_block.py
  * screening: drop (snapshot, antenna) vectors with estimated CSI error > -30 dB
  * fit LMMSE statistics on runs 0053-0055 (all antennas)
  * validation run 0056 is cut into blocks of 50 consecutive snapshots:
      even blocks -> choose the LMMSE diagonal loading, odd blocks -> reported numbers
  * test run 0057 is NOT used here
  * one global scale from the training data; pilot SNR relative to average power
  * NMSE (dB) on withheld subcarriers, 95% bootstrap CI over the reported blocks

    python experiments/01_baselines_fulldata/run_baselines.py --member <name>
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from estimators import (LMMSE, add_noise, channel_stats, linear_matrix, nmse_db,  # noqa: E402
                        pilot_positions, spline_matrix, withheld)
from runlog import save_run  # noqa: E402

SNRS = [0, 5, 10, 15, 20, 25, 30]
DENSITIES = [16, 8, 4]
RHOS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
METHODS = {"linear": ("Linear interpolation", "#2a78d6", "o"),
           "spline": ("Cubic spline", "#eb6834", "s"),
           "lmmse": ("LMMSE (training statistics)", "#1baf7a", "D")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "processed", "dichasus005x_sc256-319.npz"))
    ap.add_argument("--member", required=True)
    ap.add_argument("--fit-runs", nargs="*", type=int, default=[53, 54, 55])
    ap.add_argument("--val-run", type=int, default=56)
    ap.add_argument("--threshold", type=float, default=-30.0)
    ap.add_argument("--block-len", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="code test on a single-run file: first 70%% fit, rest validation; not saved to results/")
    a = ap.parse_args()
    t_start = time.time()

    z = np.load(a.data)
    csi, quality, run, index = z["csi"], z["quality"], z["run"], z["index"]
    n_ant, n = csi.shape[1], csi.shape[2]
    if a.smoke:
        cut = np.quantile(index, 0.7)
        fit_snap, val_snap = index < cut, index >= cut
    else:
        fit_snap, val_snap = np.isin(run, a.fit_runs), run == a.val_run
    keep = quality <= a.threshold                                   # (N, 32)

    def vectors(snap_mask):
        m = snap_mask[:, None] & keep
        snap_id = np.broadcast_to(np.arange(len(run))[:, None], keep.shape)[m]
        return csi[m].astype(np.complex128), snap_id

    Hf, _ = vectors(fit_snap)
    Hv, sid = vectors(val_snap)
    block = index[sid] // a.block_len
    sel, rep = block % 2 == 0, block % 2 == 1
    rep_blocks, rep_gid = np.unique(block[rep], return_inverse=True)

    scale = 1 / np.sqrt(np.mean(np.abs(Hf) ** 2))
    Hf *= scale
    Hv *= scale
    mu, C = channel_stats(Hf)

    summary = {
        "fit_vectors": int(len(Hf)), "val_vectors_select": int(sel.sum()), "val_vectors_report": int(rep.sum()),
        "report_blocks": int(len(rep_blocks)),
        "screened_fraction_by_run": {int(r): round(float(np.mean(quality[run == r] > a.threshold)), 5)
                                     for r in np.unique(run)},
        "quality_db_percentiles_1_50_99": np.round(np.percentile(quality, [1, 50, 99]), 2).tolist(),
    }
    print("data:", summary, flush=True)

    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(rep_blocks), (2000, len(rep_blocks)))
    metrics = {}
    for n_obs in DENSITIES:
        p = pilot_positions(n, n_obs)
        miss = withheld(n, p)
        Wl, Ws = linear_matrix(n, p), spline_matrix(n, p)
        lm = LMMSE(mu, C, p)
        metrics[str(n_obs)] = {}
        for snr in SNRS:
            nv = 10 ** (-snr / 10)
            Y = add_noise(Hv[:, p], nv, seed=1000 * n_obs + snr)
            Ys, Yr, Hs, Hr = Y[sel], Y[rep], Hv[sel], Hv[rep]
            rho = min(RHOS, key=lambda r: nmse_db(lm.apply(Ys, lm.matrix(nv, r))[:, miss], Hs[:, miss]))
            est = {"linear": Yr @ Wl.T, "spline": Yr @ Ws.T, "lmmse": lm.apply(Yr, lm.matrix(nv, rho))}
            row = {"lmmse_rho": rho}
            ref_b = np.bincount(rep_gid, np.sum(np.abs(Hr[:, miss]) ** 2, 1), len(rep_blocks))
            ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
            bs = {}
            for k, e in est.items():
                err_b = np.bincount(rep_gid, np.sum(np.abs(e[:, miss] - Hr[:, miss]) ** 2, 1), len(rep_blocks))
                bs[k] = 10 * np.log10(err_b[boots].sum(1) / ref_b[boots].sum(1))
                row[k] = {"nmse_db": round(nmse_db(e[:, miss], Hr[:, miss]), 3), "ci95": ci(bs[k])}
            # paired: the same resampled blocks for both methods, so position effects cancel
            for other in ("linear", "spline"):
                row[f"gain_lmmse_over_{other}_db"] = {
                    "value": round(row[other]["nmse_db"] - row["lmmse"]["nmse_db"], 3),
                    "ci95": ci(bs[other] - bs["lmmse"])}
            metrics[str(n_obs)][str(snr)] = row
            print(f"  {n_obs:2d}/{n} pilots, {snr:2d} dB: " +
                  "  ".join(f"{k} {row[k]['nmse_db']:6.2f}" for k in METHODS) + f"  (rho {rho:g})", flush=True)

    # the figure is written to a temporary folder and copied into results/ by save_run, so a run
    # never modifies tracked files (which would mark the record as "uncommitted code")
    fig_dir = tempfile.mkdtemp(prefix="baselines_")
    fig_path = os.path.join(fig_dir, "nmse_vs_snr_val.png")
    plot(metrics, summary, fig_path, a.smoke)
    try:
        data_ref = os.path.relpath(a.data, ROOT).replace(os.sep, "/")
    except ValueError:  # different drive
        data_ref = os.path.basename(a.data)
    config = {"data": data_ref, "block": z["block"].tolist(),
              "fit_runs": a.fit_runs, "val_run": a.val_run, "threshold_db": a.threshold,
              "block_len": a.block_len, "snrs_db": SNRS, "densities": DENSITIES, "rho_grid": RHOS,
              "scale": float(scale), "smoke": a.smoke, "runtime_s": round(time.time() - t_start, 1)}
    if a.smoke:
        print(f"smoke test finished (nothing saved to results/); figure: {fig_path}")
        return
    save_run("baselines-val", a.member, config, {"summary": summary, "nmse": metrics}, files=[fig_path],
             note="full-data baselines on validation run 0056 (odd 50-snapshot blocks); test run 0057 untouched")


def plot(metrics, summary, path, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid, surface = "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, len(DENSITIES), figsize=(14, 4.6), sharey=True, facecolor=surface)
    for ax, n_obs in zip(axes, DENSITIES):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for k, (name, col, mk) in METHODS.items():
            rows = [metrics[str(n_obs)][str(s)][k] for s in SNRS]
            y = np.array([r["nmse_db"] for r in rows])
            lo = y - np.array([r["ci95"][0] for r in rows])
            hi = np.array([r["ci95"][1] for r in rows]) - y
            ax.errorbar(SNRS, y, yerr=[lo, hi], color=col, linewidth=2, elinewidth=1, capsize=3,
                        marker=mk, markersize=6, markeredgecolor=surface, markeredgewidth=1.2, label=name)
        ax.set_title(f"{n_obs} of 64 pilots", loc="left", fontweight="bold")
        ax.set_xlabel("Pilot SNR (dB)", color=ink2)
    axes[0].set_ylabel("NMSE on withheld subcarriers (dB)", color=ink2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink2, loc="lower left")
    if smoke:
        note = (f"SMOKE TEST on one run: fit on its first 70% ({summary['fit_vectors']} vectors), "
                f"reported on odd blocks of the rest ({summary['val_vectors_report']} vectors).")
    else:
        note = (f"Fit: runs 0053-0055 ({summary['fit_vectors']} vectors). Reported: validation run 0056, "
                f"odd 50-snapshot blocks ({summary['val_vectors_report']} vectors). Test run 0057 not used.\n"
                "Error bars: 95% bootstrap CI of each NMSE over blocks (they mostly reflect how the error level "
                "varies with position); paired CIs of the gaps between methods are in run.json.")
    fig.text(0.01, 0.01, note, fontsize=8.5, color=ink2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
