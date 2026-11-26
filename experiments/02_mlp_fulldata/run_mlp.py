"""Full-data neural estimators versus LMMSE on validation run 0056 (same protocol as 01).

Variants (all take the noisy pilots + pilot SNR and output the withheld subcarriers):
  mlp             2x128 ReLU, plain MSE                       (the proposal's model)
  mlp_relweight   2x128, each sample's loss divided by the LMMSE error expected at its SNR
                  (so high-SNR samples are not drowned out by low-SNR ones)
  residual        2x128 that only learns a correction to the LMMSE estimate
  mlp_wide        3x256, plain MSE (is capacity the limit?)
  mlp_blind       2x128, plain MSE, NO SNR input (the noise-mismatch experiment needs it)
Training: pilot SNR ~ U[0,30] dB with fresh noise every batch, Adam, early stopping on the even
validation blocks; reported on the odd blocks with the SAME noise realizations as experiment 01.
The best weights of every model are saved (data/runs/<run folder>/<pilots>_<variant>.npz, SHA-256
in run.json), so later experiments evaluate exactly these models without retraining.

    python experiments/02_mlp_fulldata/run_mlp.py --member <name> --densities 16 8 4
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # default thread count is far slower on small matrices
import argparse  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import DEFAULT_NPZ, bootstrap_ci, load_split  # noqa: E402
from estimators import (LMMSE, add_noise, channel_stats, linear_matrix, nmse_db,  # noqa: E402
                        pilot_positions, withheld)
from mlp_numpy import MLP  # noqa: E402
from neural import VARIANTS as SPECS, features, load_predictor, make_predictor  # noqa: E402
from runlog import save_run  # noqa: E402

SNRS = [0, 5, 10, 15, 20, 25, 30]
VAL_SNRS = [0, 10, 20, 30]
RHOS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
VARIANTS = {"mlp": ("MLP 2x128 (proposal)", "#2a78d6", "o"),           # legend name, colour, marker
            "mlp_relweight": ("MLP 2x128, relative-error weighting", "#eb6834", "s"),
            "residual": ("LMMSE + residual MLP 2x128", "#1baf7a", "D"),
            "mlp_wide": ("MLP 3x256", "#eda100", "^"),
            "mlp_blind": ("MLP 2x128, no SNR input", "#7b61c9", "v")}


def lmmse_expected_mse(lm, miss, nv):
    """Per-sample expected LMMSE error (sum over withheld subcarriers) for noise variances nv."""
    lam, U, A = lm._eig
    Am = np.abs(A[miss]) ** 2                          # (|miss|, |p|)
    tr = np.real(np.trace(lm.C[np.ix_(miss, miss)]))
    return tr - (Am.sum(0)[None, :] / (lam[None, :] + nv[:, None])).sum(1)


def train(variant, d, p, miss, lm, epochs, batch, seed, log):
    hidden, snr_input, _ = SPECS[variant]
    rng = np.random.default_rng(seed)
    net = MLP([2 * len(p) + 1] + hidden + [2 * len(miss)], seed=seed)
    Hf, Hs = d.Hf, d.Hv[d.sel]
    Ysel = {s: add_noise(Hs[:, p], 10 ** (-s / 10), seed=7000 + s) for s in VAL_SNRS}
    lm.apply_per_sample(Hs[:1, p], 1.0)                 # builds the eigen-decomposition
    predict = make_predictor(variant, net, lm, miss)

    steps = len(Hf) // batch
    best, best_state, t0 = np.inf, None, time.time()
    for ep in range(epochs):
        lr = 1e-3 if ep < 0.6 * epochs else (3e-4 if ep < 0.85 * epochs else 1e-4)
        perm = rng.permutation(len(Hf))
        for b in range(steps):
            h = Hf[perm[b * batch:(b + 1) * batch]]
            s = rng.uniform(0, 30, len(h))
            nv = 10 ** (-s / 10)
            Y = h[:, p] + np.sqrt(nv / 2)[:, None] * (rng.standard_normal((len(h), len(p)))
                                                     + 1j * rng.standard_normal((len(h), len(p))))
            T = h[:, miss]
            if variant == "residual":
                T = T - lm.apply_per_sample(Y, nv)[:, miss]
            out, cache = net.forward(features(Y, s, snr_input))
            diff = out - np.concatenate([T.real, T.imag], axis=1)
            if variant == "mlp_relweight":
                w = 1 / lmmse_expected_mse(lm, miss, nv)
                diff = diff * (w / w.mean())[:, None]
            net.step(cache, 2 * diff / diff.size, lr)
        val = float(np.mean([nmse_db(predict(Ysel[s], np.full(len(Hs), s)), Hs[:, miss]) for s in VAL_SNRS]))
        if val < best:
            best, best_state = val, net.state()
        log(f"    {variant} epoch {ep + 1}/{epochs}: val {val:.3f} dB  ({time.time() - t0:.0f} s)")
    net.load(best_state)
    return net, predict, {"best_val_db": round(best, 3), "params": net.n_params(),
                          "train_s": round(time.time() - t0, 1)}


def save_and_check(net, predict, path, lm, miss, Ycheck, meta):
    """Save the weights, reload them and confirm the reloaded model predicts exactly the same."""
    net.save(path, **meta)
    again, _ = load_predictor(path, lm, miss)
    s = np.linspace(0, 30, len(Ycheck))
    if not np.array_equal(again(Ycheck, s), predict(Ycheck, s)):
        raise RuntimeError(f"reloaded model {path} does not reproduce the trained one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_NPZ)
    ap.add_argument("--member", required=True)
    ap.add_argument("--densities", nargs="*", type=int, default=[16, 8])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    t_start = time.time()

    d = load_split(a.data, smoke=a.smoke)
    log(f"fit vectors {len(d.Hf)}, select {int(d.sel.sum())}, report {int(d.rep.sum())}")
    mu, C = channel_stats(d.Hf)
    Hs, Hr = d.Hv[d.sel], d.Hv[d.rep]
    boots = np.random.default_rng(0).integers(0, d.n_rep_blocks, (2000, d.n_rep_blocks))
    metrics, train_info = {}, {}
    weight_dir, artifacts = tempfile.mkdtemp(prefix="mlp_weights_"), []
    for n_obs in a.densities:
        p = pilot_positions(d.n, n_obs)
        miss = withheld(d.n, p)
        lm = LMMSE(mu, C, p)
        models = {}
        for v in a.variants:
            log(f"  training {v} for {n_obs} pilots")
            net, models[v], train_info[f"{n_obs}_{v}"] = train(v, d, p, miss, lm, a.epochs, a.batch, a.seed, log)
            path = os.path.join(weight_dir, f"{n_obs}_{v}.npz")
            save_and_check(net, models[v], path, lm, miss, Hs[:512, p],
                           dict(variant=v, n_obs=n_obs, pilots=p.tolist(), withheld=miss.tolist(), seed=a.seed,
                                epochs=a.epochs, batch=a.batch, block=d.block, threshold_db=d.threshold,
                                scale=d.scale, best_val_db=train_info[f"{n_obs}_{v}"]["best_val_db"]))
            artifacts.append(path)
        metrics[str(n_obs)] = {}
        Wl = linear_matrix(d.n, p)
        for snr in SNRS:
            nv = 10 ** (-snr / 10)
            Y = add_noise(d.Hv[:, p], nv, seed=1000 * n_obs + snr)      # identical to experiment 01
            Ys, Yr = Y[d.sel], Y[d.rep]
            rho = min(RHOS, key=lambda r: nmse_db(lm.apply(Ys, lm.matrix(nv, r))[:, miss], Hs[:, miss]))
            est = {"linear": (Yr @ Wl.T)[:, miss], "lmmse": lm.apply(Yr, lm.matrix(nv, rho))[:, miss]}
            for v in a.variants:
                est[v] = models[v](Yr, np.full(len(Yr), snr))
            ref_b = np.bincount(d.rep_gid, np.sum(np.abs(Hr[:, miss]) ** 2, 1), d.n_rep_blocks)
            row, bs = {}, {}
            for k, e in est.items():
                err_b = np.bincount(d.rep_gid, np.sum(np.abs(e - Hr[:, miss]) ** 2, 1), d.n_rep_blocks)
                bs[k], ci = bootstrap_ci(err_b, ref_b, boots)
                row[k] = {"nmse_db": round(nmse_db(e, Hr[:, miss]), 3), "ci95": ci}
            for v in a.variants:  # positive = better than LMMSE
                diff = bs["lmmse"] - bs[v]
                row[f"{v}_gain_over_lmmse_db"] = {
                    "value": round(row["lmmse"]["nmse_db"] - row[v]["nmse_db"], 3),
                    "ci95": [round(float(np.percentile(diff, 2.5)), 3), round(float(np.percentile(diff, 97.5)), 3)]}
            metrics[str(n_obs)][str(snr)] = row
            log(f"  {n_obs:2d} pilots {snr:2d} dB: linear {row['linear']['nmse_db']:6.2f} lmmse {row['lmmse']['nmse_db']:6.2f} "
                + " ".join(f"{v} {row[v]['nmse_db']:6.2f}" for v in a.variants))

    fig_path = os.path.join(tempfile.mkdtemp(prefix="mlp_"), "gap_to_lmmse_val.png")
    files = []
    try:  # a plotting problem must never lose the metrics of a long run
        plot(metrics, a, fig_path, len(d.Hf))
        files = [fig_path]
    except Exception as e:
        log(f"plot failed ({e}); metrics are still saved")
    config = {"block": d.block, "threshold_db": d.threshold, "block_len": d.block_len, "densities": a.densities,
              "variants": {v: SPECS[v][0] for v in a.variants},
              "snr_input": {v: SPECS[v][1] for v in a.variants}, "epochs": a.epochs, "batch": a.batch,
              "seed": a.seed, "snr_train": "U[0,30] dB", "snrs_db": SNRS, "smoke": a.smoke,
              "runtime_s": round(time.time() - t_start, 1)}
    if a.smoke:
        log(f"smoke test finished; figure {fig_path}; weights {weight_dir}")
        return
    save_run("mlp-val", a.member, config, {"nmse": metrics, "training": train_info}, files=files,
             artifacts=artifacts,
             note="neural estimators vs LMMSE on validation run 0056 (odd blocks); test run 0057 untouched")


def plot(metrics, a, path, n_fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid, surface = "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, len(a.densities), figsize=(6.2 * len(a.densities), 4.6), sharey=True,
                             facecolor=surface, squeeze=False)
    for ax, n_obs in zip(axes[0], a.densities):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axhline(0, color=ink2, linewidth=1.2)
        ax.text(SNRS[0], 0.08, "LMMSE", color=ink2, fontsize=9, va="bottom")
        for v in a.variants:
            name, col, mk = VARIANTS[v]
            g = [metrics[str(n_obs)][str(s)][f"{v}_gain_over_lmmse_db"] for s in SNRS]
            y = np.array([x["value"] for x in g])
            err = [np.maximum(y - np.array([x["ci95"][0] for x in g]), 0),   # rounding can give -0.001
                   np.maximum(np.array([x["ci95"][1] for x in g]) - y, 0)]
            ax.errorbar(SNRS, y, yerr=err, color=col, linewidth=2, elinewidth=1, capsize=3, marker=mk,
                        markersize=6, markeredgecolor=surface, markeredgewidth=1.2, label=name)
        ax.set_title(f"{n_obs} of 64 pilots", loc="left", fontweight="bold")
        ax.set_xlabel("Pilot SNR (dB)", color=ink2)
    axes[0][0].set_ylabel("Gain over LMMSE (dB, positive = better)", color=ink2)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=ink2, loc="lower left")
    fig.text(0.01, 0.01, f"Trained on runs 0053-0055 ({n_fit} vectors), pilot SNR U[0,30] dB; reported on validation "
             "run 0056 (odd 50-snapshot blocks). Error bars: paired 95% bootstrap CI. Test run 0057 not used.",
             fontsize=8.5, color=ink2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
