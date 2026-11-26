"""Noise-power (SNR) mismatch: how much do LMMSE and the trained MLPs lose when the pilot SNR they
assume is wrong? Validation run 0056 only, same protocol and noise realisations as experiments 01/02.

Grid per pilot density: true SNR t in {0,5,...,30} x assumed SNR a in {0,5,...,30}.
  Y = add_noise(Hv[:, p], 10**(-t/10), seed=1000*n_obs+t)     (so the diagonal t = a reproduces 01/02)
  linear          needs no SNR
  lmmse           designed for the ASSUMED SNR: G = lm.matrix(10**(-a/10), rho_a), with rho_a chosen on
                  the even blocks under matched noise at a, exactly as in experiment 01
  mlp, mlp_relweight, residual (any SNR-aware variant)
                  get the assumed SNR as input; the residual's LMMSE base also uses it
  mlp_blind       has no SNR input, so it does not depend on a: a separate method
Fairness rule: LMMSE and the SNR-aware MLPs receive the SAME wrong SNR. Every design (weights, rho)
is frozen per assumed SNR; the true noise only changes the input.

Models are the weights saved by experiments/02: one results folder per seed (--runs, SHA-256 checked
by runlog.load_artifacts), or plain folders of MLP.save files (--weights-dir, for smoke tests).
Reported on the odd 50-snapshot blocks. MLP numbers without a seed are seed averages: the per-block
squared errors are averaged over seeds, then turned into NMSE. Gains over LMMSE at the same (t, a)
get a paired block bootstrap (B = 1000, the same resampled blocks for both methods) and the per-seed
min / max. The test run 0057 is never used (dataset.load_split refuses it as a fitting or validation run).

    python experiments/04_snr_mismatch/run_mismatch.py --member <name> --runs results/<seed0 run> ...
    python experiments/04_snr_mismatch/run_mismatch.py --member <name> --smoke --weights-dir <dir> ...
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # default thread count is far slower on small matrices
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import DEFAULT_NPZ, load_split  # noqa: E402
from estimators import (LMMSE, add_noise, channel_stats, linear_matrix, nmse_db,  # noqa: E402
                        pilot_positions, withheld)
from neural import VARIANTS as SPECS, load_predictor  # noqa: E402
from runlog import load_artifacts, save_run, sha256  # noqa: E402

SNRS = [0, 5, 10, 15, 20, 25, 30]
SMOKE_SNRS = [0, 10, 20, 30]                       # coarser grid, so the smoke test stays fast
RHOS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]        # LMMSE loading grid of experiments 01 / 02
N_BOOT = 1000
FIXED_DESIGNS = [10, 20]                           # receivers designed for one SNR, used at every true SNR
NEAR_OFFSET = 10                                   # worst case also reported for |assumed - true| <= 10 dB
FIG1_METHODS = ["lmmse", "mlp", "residual", "mlp_relweight"]
FIG2_DESIGN = 20


def log(m):
    print(m, flush=True)


def r3(x):
    return np.round(np.asarray(x, float), 3).tolist()


def shown(path):
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:  # different drive
        return path.replace(os.sep, "/")


# ------------------------------------------------------------------------------------- models
def resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(path) if os.path.exists(path) else os.path.join(ROOT, path)


def run_artifacts(run_dir):
    """(run.json record, {file name: path}) of an experiments/02 run, every SHA-256 checked."""
    try:
        return load_artifacts(run_dir)
    except KeyError:
        raise ValueError(f"{run_dir}: run.json lists no saved weights (run before weights were saved); "
                         "retrain with experiments/02_mlp_fulldata/run_mlp.py") from None
    except FileNotFoundError:
        if not os.path.exists(os.path.join(run_dir, "run.json")):
            raise
    # a results folder of another checkout keeps its weights in THAT checkout's data/runs/
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        rec = json.load(f)
    art_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(run_dir))), rec["artifacts_dir"])
    paths = {}
    for base, info in rec["artifacts"].items():
        p = os.path.join(art_dir, base)
        if sha256(p) != info["sha256"]:
            raise ValueError(f"{p} does not match the SHA-256 recorded in {run_dir}/run.json")
        paths[base] = p
    return rec, paths


def read_meta(path):
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def find_models(a):
    """{(n_obs, variant): {seed: model info}}; every source file is identified by its SHA-256."""
    sources = []
    if a.runs:
        for r in a.runs:
            rd = resolve(r)
            rec, paths = run_artifacts(rd)
            if rec.get("git_uncommitted_changes"):
                log(f"warning: {shown(rd)} was trained with uncommitted code changes")
            for base, p in sorted(paths.items()):
                if base.endswith(".npz"):
                    sources.append((p, {"run": shown(rd), "file": base, "sha256": rec["artifacts"][base]["sha256"],
                                        "git_commit": rec.get("git_commit")}, rec))
    else:
        for wd in a.weights_dir:
            files = sorted(glob.glob(os.path.join(resolve(wd), "*.npz")))
            if not files:
                raise ValueError(f"no .npz weights in {wd}")
            for p in files:
                sources.append((p, {"dir": shown(resolve(wd)), "file": os.path.basename(p), "sha256": sha256(p)},
                                None))
    models = {}
    for p, prov, rec in sources:
        meta = read_meta(p)
        key, seed = (int(meta["n_obs"]), str(meta["variant"])), int(meta["seed"])
        if key[1] not in SPECS:
            raise ValueError(f"{p}: unknown variant {key[1]!r}")
        if (a.densities and key[0] not in a.densities) or (a.variants and key[1] not in a.variants):
            continue
        slot = models.setdefault(key, {})
        if seed in slot:
            raise ValueError(f"two models for {key[0]} pilots / {key[1]} / seed {seed}: "
                             f"{slot[seed]['path']} and {p}")
        slot[seed] = {"path": p, "meta": meta, "prov": dict(prov, seed=seed), "record": rec}
    if not models:
        raise ValueError("no models left after the --densities / --variants filter")
    return models


def check_models(models, d, smoke):
    """The weights must come from the same data and protocol: same global scale, block and screening."""
    bad = []
    for (n_obs, v), slot in sorted(models.items()):
        for seed, m in sorted(slot.items()):
            meta = m["meta"]
            why = []
            if not np.isclose(float(meta["scale"]), d.scale, rtol=1e-9, atol=0):
                why.append(f"scale {meta['scale']} != {d.scale}")
            if "block" in meta and list(meta["block"]) != list(d.block):
                why.append(f"block {meta['block']} != {d.block}")
            if "threshold_db" in meta and float(meta["threshold_db"]) != float(d.threshold):
                why.append(f"threshold {meta['threshold_db']} != {d.threshold}")
            if why:
                bad.append(f"{m['path']}: " + "; ".join(why))
    if bad and not smoke:
        raise ValueError("weights do not match this data:\n  " + "\n  ".join(bad))
    if bad:
        log(f"smoke: ignoring {len(bad)} data mismatches of the weights (expected: smoke data), e.g. {bad[0]}")
    missing = {k: sorted(set().union(*models.values()) - set(s)) for k, s in models.items()}
    for k, miss_seeds in sorted(missing.items()):
        if miss_seeds:
            log(f"warning: {k[0]} pilots / {k[1]} has no model for seeds {miss_seeds}; its seed average "
                "uses the seeds it has")


# ----------------------------------------------------------------------------------- evaluate
def evaluate_density(n_obs, models_n, d, mu, C, snrs):
    """Per-seed NMSE and per-block squared errors on the odd blocks for the whole (true, assumed) grid."""
    t0 = time.time()
    p = pilot_positions(d.n, n_obs)
    miss = withheld(d.n, p)
    lm = LMMSE(mu, C, p)                                     # one object: the residuals share its eigen cache
    Wl = linear_matrix(d.n, p)
    Hs, Hr = d.Hv[d.sel][:, miss], d.Hv[d.rep][:, miss]
    gid, nb = d.rep_gid, d.n_rep_blocks
    ref_tot = np.sum(np.abs(Hr) ** 2)
    S = len(snrs)

    # design per ASSUMED SNR, chosen under matched noise on the even blocks (identical to experiment 01)
    rho, G = {}, {}
    for s in snrs:
        nv = 10 ** (-s / 10)
        Ys = add_noise(d.Hv[:, p], nv, seed=1000 * n_obs + s)[d.sel]
        rho[s] = min(RHOS, key=lambda r: nmse_db(lm.apply(Ys, lm.matrix(nv, r))[:, miss], Hs))
        G[s] = lm.matrix(nv, rho[s])
    del Ys

    nets = {}
    for v, slot in sorted(models_n.items()):
        _, aware, residual = SPECS[v]
        if residual and not aware:      # its LMMSE base would silently run at the dummy 0 dB below
            raise ValueError(f"{v}: a residual variant without SNR input is not supported here")
        seeds = sorted(slot)
        preds = [load_predictor(slot[s]["path"], lm, miss)[0] for s in seeds]
        shape = (len(seeds), S, S) if aware else (len(seeds), S)
        nets[v] = {"aware": aware, "seeds": seeds, "preds": preds,
                   "nmse": np.zeros(shape), "err": np.zeros(shape + (nb,))}
    raw = {"n_obs": n_obs, "pilots": p.tolist(), "rho": rho,
           "ref_b": np.bincount(gid, np.sum(np.abs(Hr) ** 2, 1), nb),
           "linear": {"nmse": np.zeros(S), "err": np.zeros((S, nb))},
           "lmmse": {"nmse": np.zeros((S, S)), "err": np.zeros((S, S, nb))}, "nets": nets}

    def put(slot, idx, est):
        sq = np.abs(est - Hr) ** 2
        slot["nmse"][idx] = 10 * np.log10(sq.sum() / ref_tot)        # bit-identical to nmse_db
        slot["err"][idx] = np.bincount(gid, sq.sum(1), nb)

    for i, t in enumerate(snrs):
        Yr = add_noise(d.Hv[:, p], 10 ** (-t / 10), seed=1000 * n_obs + t)[d.rep]   # only odd blocks evaluated
        put(raw["linear"], i, (Yr @ Wl.T)[:, miss])
        for v, net in nets.items():
            if not net["aware"]:
                for k, pred in enumerate(net["preds"]):
                    put(net, (k, i), pred(Yr, np.zeros(len(Yr))))     # blind: the SNR input is ignored
        for j, s in enumerate(snrs):
            put(raw["lmmse"], (i, j), lm.apply(Yr, G[s])[:, miss])
            s_vec = np.full(len(Yr), float(s))
            for v, net in nets.items():
                if net["aware"]:
                    for k, pred in enumerate(net["preds"]):
                        put(net, (k, i, j), pred(Yr, s_vec))
        line = f"lmmse {raw['lmmse']['nmse'][i, i]:7.2f}"
        for v, net in nets.items():
            line += f"  {v} {np.mean(net['nmse'][:, i, i] if net['aware'] else net['nmse'][:, i]):7.2f}"
        log(f"  {n_obs:2d} pilots, true {t:2d} dB (matched): {line}   ({time.time() - t0:.0f} s)")
    for net in nets.values():
        del net["preds"]
    raw["runtime_s"] = round(time.time() - t0, 1)
    return raw


# ---------------------------------------------------------------------------------- summaries
def offset_view(N, ref_diag, snrs):
    """Mean over true SNRs of N[t, t+o] - ref_diag[t], for every offset o with t+o on the grid."""
    offs = sorted({s - t for t in snrs for s in snrs})
    mean, count = [], []
    for o in offs:
        pairs = [(i, snrs.index(t + o)) for i, t in enumerate(snrs) if t + o in snrs]
        mean.append(float(np.mean([N[i, j] - ref_diag[i] for i, j in pairs])))
        count.append(len(pairs))
    return offs, mean, count


def worst(L, snrs, max_offset=None):
    """Largest loss on the grid L[true, assumed] (optionally only |assumed - true| <= max_offset)."""
    t, s = np.meshgrid(snrs, snrs, indexing="ij")
    Lm = np.where(np.abs(s - t) <= max_offset, L, -np.inf) if max_offset is not None else L
    i, j = np.unravel_index(np.argmax(Lm), Lm.shape)
    return {"loss_db": round(float(L[i, j]), 3), "true_snr_db": int(snrs[i]), "assumed_snr_db": int(snrs[j])}


def summarize(raw, snrs, boots, ci_digits=3):
    """All reported numbers of one density, from the per-seed NMSE and per-block errors. Gain CI bounds are
    rounded to ci_digits decimals; gain_ci_excludes_zero (+1 above 0, -1 below 0, 0 otherwise) comes from the
    unrounded percentiles."""
    S = len(snrs)
    rd = lambda x: np.round(np.asarray(x, float), ci_digits).tolist()
    ref_b = raw["ref_b"]
    bref = ref_b[boots].sum(-1)

    def boot_db(err):                                  # (..., n_blocks) -> (..., B)
        return 10 * np.log10(err[..., boots].sum(-1) / bref)

    lm_n = raw["lmmse"]["nmse"]
    lm_boot = boot_db(raw["lmmse"]["err"])
    diag = np.diag(lm_n)
    grid = {"pilots": raw["pilots"], "lmmse_rho": {str(s): raw["rho"][s] for s in snrs},
            "lmmse": {"nmse_db": r3(lm_n)}, "linear": {"nmse_db": r3(raw["linear"]["nmse"])}}
    curves = {"lmmse": lm_n, "linear": raw["linear"]["nmse"]}   # seed-averaged NMSE, (S, S) or (S,)
    for v, net in raw["nets"].items():
        ens = net["err"].mean(0)
        ens_db = 10 * np.log10(ens.sum(-1) / ref_b.sum())
        curves[v] = ens_db
        if net["aware"]:
            gain, diff, seed_gain = lm_n - ens_db, lm_boot - boot_db(ens), lm_n[None] - net["nmse"]
        else:   # compared with the LMMSE designed for each assumed SNR
            gain = lm_n - ens_db[:, None]
            diff = lm_boot - boot_db(ens)[:, None, :]
            seed_gain = lm_n[None] - net["nmse"][:, :, None]
        lo, hi = np.percentile(diff, [2.5, 97.5], axis=-1)
        grid[v] = {"seeds": net["seeds"], "snr_input": net["aware"], "nmse_db": r3(ens_db),
                   "per_seed_nmse_db": {str(s): r3(x) for s, x in zip(net["seeds"], net["nmse"])},
                   "gain_over_lmmse_db": r3(gain), "gain_ci95_low": rd(lo), "gain_ci95_high": rd(hi),
                   "gain_ci_excludes_zero": np.where(lo > 0, 1, np.where(hi < 0, -1, 0)).tolist(),
                   "gain_seed_min": r3(seed_gain.min(0)), "gain_seed_max": r3(seed_gain.max(0))}

    ii = np.arange(S)
    matched = {"lmmse": {"nmse_db": r3(diag)}, "linear": {"nmse_db": r3(raw["linear"]["nmse"])}}
    for v, g in grid.items():
        if v in ("pilots", "lmmse_rho", "lmmse", "linear"):
            continue
        G = {k: np.asarray(g[k]) for k in ("gain_over_lmmse_db", "gain_ci95_low", "gain_ci95_high",
                                           "gain_seed_min", "gain_seed_max")}
        n = np.asarray(g["nmse_db"])
        matched[v] = {"nmse_db": (np.diag(n) if g["snr_input"] else n).tolist(),
                      "gain_over_lmmse_db": G["gain_over_lmmse_db"][ii, ii].tolist(),
                      "gain_ci95": np.stack([G["gain_ci95_low"][ii, ii], G["gain_ci95_high"][ii, ii]], 1).tolist(),
                      "gain_ci_excludes_zero": np.asarray(g["gain_ci_excludes_zero"])[ii, ii].tolist(),
                      "gain_seed_range": np.stack([G["gain_seed_min"][ii, ii], G["gain_seed_max"][ii, ii]], 1).tolist()}

    offsets = {"vs_own_matched": {}, "vs_matched_lmmse": {}}
    worst_case, by_true = {}, {}
    for m, N in curves.items():
        if N.ndim == 2:     # depends on the assumed SNR
            offs, own, count = offset_view(N, np.diag(N), snrs)
            _, gen, _ = offset_view(N, diag, snrs)
            offsets["vs_own_matched"][m], offsets["vs_matched_lmmse"][m] = r3(own), r3(gen)
            L_own, L_gen = N - np.diag(N)[:, None], N - diag[:, None]
            worst_case[m] = {"vs_own_matched": worst(L_own, snrs),
                             f"vs_own_matched_within_{NEAR_OFFSET}db": worst(L_own, snrs, NEAR_OFFSET),
                             "vs_matched_lmmse": worst(L_gen, snrs),
                             f"vs_matched_lmmse_within_{NEAR_OFFSET}db": worst(L_gen, snrs, NEAR_OFFSET)}
        else:               # SNR-free: no own mismatch loss; its price is the gap to the matched LMMSE
            L = N - diag
            by_true[m] = r3(L)
            k = int(np.argmax(L))
            worst_case[m] = {"vs_own_matched": None,
                             "vs_matched_lmmse": {"loss_db": round(float(L[k]), 3), "true_snr_db": int(snrs[k]),
                                                  "assumed_snr_db": None}}
    offsets.update(offsets_db=offs, n_true=count, snr_free_vs_matched_lmmse_by_true=by_true)

    fixed = {}
    for s in FIXED_DESIGNS:
        if s not in snrs:
            continue
        j = snrs.index(s)
        fixed[str(s)] = {m: r3(N[:, j] if N.ndim == 2 else N) for m, N in curves.items()}
        fixed[str(s)]["lmmse_matched"] = r3(diag)
    return {"grid": grid, "matched": matched, "offset_loss_db": offsets, "worst_case_loss_db": worst_case,
            "fixed_design": fixed}


def check_vs_02(raws, models, snrs):
    """Matched diagonal vs the numbers experiment 02 reported for the same weights (needs --runs)."""
    diffs = []
    for n_obs, raw in raws.items():
        recs = {}
        for (n, v), slot in models.items():
            if n != n_obs:
                continue
            for seed, m in slot.items():
                if m["record"] is None:
                    continue
                recs[seed] = m["record"]
                net = raw["nets"][v]
                k = net["seeds"].index(seed)
                for i, t in enumerate(snrs):
                    old = m["record"].get("metrics", {}).get("nmse", {}).get(str(n_obs), {}).get(str(t), {}).get(v)
                    if old is not None:
                        new = net["nmse"][k, i, i] if net["aware"] else net["nmse"][k, i]
                        diffs.append((abs(round(float(new), 3) - old["nmse_db"]), f"{n_obs}/{v}/seed{seed}/{t}dB"))
        for rec in recs.values():
            for i, t in enumerate(snrs):
                row = rec.get("metrics", {}).get("nmse", {}).get(str(n_obs), {}).get(str(t), {})
                for meth, new in (("lmmse", raw["lmmse"]["nmse"][i, i]), ("linear", raw["linear"]["nmse"][i])):
                    if meth in row:
                        diffs.append((abs(round(float(new), 3) - row[meth]["nmse_db"]), f"{n_obs}/{meth}/{t}dB"))
    if not diffs:
        return None
    worst_d = max(diffs)
    return {"n_compared": len(diffs), "max_abs_diff_db": round(worst_d[0], 4), "worst_entry": worst_d[1]}


# ------------------------------------------------------------------------------------- figures
def plot_offset(metrics, densities, path):
    from style import COLORS, LABELS, MARKERS, PAGE_W, apply_style
    plt = apply_style("paper")
    fig, axes = plt.subplots(1, len(densities), figsize=(PAGE_W, 2.4), sharey=True, squeeze=False)
    for ax, n_obs in zip(axes[0], densities):
        ov = metrics["offset_loss_db"][str(n_obs)]
        ax.axhline(0, color="#52514e", linewidth=0.6)
        ax.axvline(0, color="#52514e", linewidth=0.6)
        for m in FIG1_METHODS:
            if m in ov["vs_own_matched"]:
                ax.plot(ov["offsets_db"], ov["vs_own_matched"][m], color=COLORS[m], marker=MARKERS[m],
                        label=LABELS[m], zorder=3 if m == "lmmse" else 4)
        ax.set_title(f"{n_obs} of 64 pilots", loc="left")
        ax.set_xlabel("Assumed − true SNR (dB)")
        ax.set_xticks(range(-30, 31, 10))
    axes[0][0].set_ylabel("NMSE loss vs. matched SNR (dB)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def plot_fixed(metrics, densities, snrs, path, design=FIG2_DESIGN):
    from style import COLORS, LABELS, MARKERS, PAGE_W, apply_style
    plt = apply_style("paper")
    fig, axes = plt.subplots(1, len(densities), figsize=(PAGE_W, 2.5), sharey=True, squeeze=False)
    lines = [("lmmse", f"LMMSE, assumes {design} dB"), ("mlp", f"MLP, assumes {design} dB"),
             ("residual", f"LMMSE + residual MLP, assumes {design} dB"), ("mlp_blind", LABELS["mlp_blind"])]
    for ax, n_obs in zip(axes[0], densities):
        fd = metrics["fixed_design"][str(n_obs)][str(design)]
        ax.axvline(design, color="#52514e", linewidth=0.6, linestyle=":")
        ax.plot(snrs, fd["lmmse_matched"], color=COLORS["genie"], linestyle="--",
                label="LMMSE, knows the true SNR", zorder=2)
        for m, lab in lines:
            if m in fd:
                ax.plot(snrs, fd[m], color=COLORS[m], marker=MARKERS[m], label=lab, zorder=3)
        ax.set_title(f"{n_obs} of 64 pilots", loc="left")
        ax.set_xlabel("True pilot SNR (dB)")
    axes[0][0].set_ylabel("NMSE on withheld subcarriers (dB)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--member", required=True)
    ap.add_argument("--data", default=DEFAULT_NPZ)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+", help="experiments/02 results folders, one per seed (weights SHA-256 checked)")
    src.add_argument("--weights-dir", nargs="+", help="plain folders of MLP.save .npz files (smoke tests)")
    ap.add_argument("--densities", nargs="*", type=int, help="default: every density with saved weights")
    ap.add_argument("--variants", nargs="*", help="default: every variant with saved weights")
    ap.add_argument("--smoke", action="store_true",
                    help="code test on ONE fitting run, SNR grid 0/10/20/30 dB; nothing saved to results/")
    a = ap.parse_args(argv)
    t_start = time.time()
    snrs = SMOKE_SNRS if a.smoke else SNRS

    models = find_models(a)                                  # fail on a wrong path before loading data
    densities = sorted({k[0] for k in models}, reverse=True)
    d = load_split(a.data, smoke=a.smoke)
    check_models(models, d, a.smoke)
    log(f"fit vectors {len(d.Hf)}, select {int(d.sel.sum())}, report {int(d.rep.sum())} in {d.n_rep_blocks} blocks; "
        f"models: " + ", ".join(f"{n}/{v} x{len(s)}" for (n, v), s in sorted(models.items(), reverse=True)))
    mu, C = channel_stats(d.Hf)                              # identical to experiment 02
    summary = {"fit_vectors": int(len(d.Hf)), "val_vectors_select": int(d.sel.sum()),
               "val_vectors_report": int(d.rep.sum()), "report_blocks": int(d.n_rep_blocks)}
    del d.Hf                                                 # ~0.5 GB not needed any more

    boots = np.random.default_rng(0).integers(0, d.n_rep_blocks, (N_BOOT, d.n_rep_blocks))
    raws, metrics = {}, {"summary": summary, "grid": {}, "matched": {}, "offset_loss_db": {},
                         "worst_case_loss_db": {}, "fixed_design": {}}
    for n_obs in densities:
        models_n = {v: slot for (n, v), slot in models.items() if n == n_obs}
        raws[n_obs] = evaluate_density(n_obs, models_n, d, mu, C, snrs)
        for k, v in summarize(raws[n_obs], snrs, boots).items():
            metrics[k][str(n_obs)] = v
        summary[f"runtime_s_{n_obs}_pilots"] = raws[n_obs]["runtime_s"]
    metrics["check_vs_experiment02"] = chk = check_vs_02(raws, models, snrs)
    if chk is not None:
        log(f"matched diagonal vs experiment 02: max |diff| {chk['max_abs_diff_db']} dB over {chk['n_compared']} "
            f"numbers ({chk['worst_entry']})" + ("  [smoke: different data, differences expected]" if a.smoke else ""))
        if chk["max_abs_diff_db"] > 0.0015 and not a.smoke:
            log("WARNING: the matched diagonal does not reproduce experiment 02; check data file and code version")
    for n_obs in densities:
        log(f"  {n_obs:2d} pilots, worst-case loss vs own matched SNR: " + "  ".join(
            f"{m} {w['vs_own_matched']['loss_db']:.2f}" for m, w in metrics["worst_case_loss_db"][str(n_obs)].items()
            if w["vs_own_matched"]))

    out_dir = tempfile.mkdtemp(prefix="snr_mismatch_")
    files = []
    try:  # a plotting problem must never lose the metrics
        f1 = os.path.join(out_dir, "offset_loss_val.png")
        plot_offset(metrics, densities, f1)
        files.append(f1)
        if str(FIG2_DESIGN) in metrics["fixed_design"][str(densities[0])]:
            f2 = os.path.join(out_dir, f"fixed_design_{FIG2_DESIGN}db_val.png")
            plot_fixed(metrics, densities, snrs, f2)
            files.append(f2)
    except Exception as e:
        log(f"plot failed ({e!r}); metrics are still saved")
    metrics["definitions"] = {
        "grid_axes": "lists are [true SNR index][assumed SNR index] over snrs_db; SNR-free methods (linear, "
                     "mlp_blind) have one index (true SNR)",
        "seed_average": "MLP numbers without a seed key: per-block squared errors averaged over seeds, then NMSE",
        "gain_over_lmmse_db": "NMSE(LMMSE designed for the assumed SNR) - NMSE(method) at the same (true, assumed); "
                              "positive = method better; CI: paired block bootstrap, B=1000",
        "offset_loss_db": "mean over true SNRs t with t+offset on the grid of NMSE(t, t+offset) - reference(t); "
                          "vs_own_matched: reference = the same method at (t, t); vs_matched_lmmse: LMMSE at (t, t)",
        "fixed_design": "NMSE vs true SNR of receivers designed for one assumed SNR; lmmse_matched knows the true SNR"}
    config = {"data": shown(os.path.abspath(a.data)), "block": d.block, "threshold_db": d.threshold,
              "block_len": d.block_len, "scale": d.scale, "snrs_db": snrs, "densities": densities,
              "variants": sorted({v for _, v in models}), "rho_grid": RHOS, "n_boot": N_BOOT, "boot_seed": 0,
              "fixed_designs_db": FIXED_DESIGNS, "near_offset_db": NEAR_OFFSET,
              "noise": "Y = add_noise(Hv[:, p], 10**(-true/10), seed=1000*n_obs+true), as experiments 01/02",
              "fairness": "LMMSE (rho chosen on even blocks at the assumed SNR) and SNR-aware MLPs get the same "
                          "assumed SNR; mlp_blind ignores it; designs frozen per assumed SNR",
              "weights": [m["prov"] for _, slot in sorted(models.items()) for _, m in sorted(slot.items())],
              "smoke": a.smoke, "runtime_s": round(time.time() - t_start, 1)}
    if a.smoke:
        with open(os.path.join(out_dir, "metrics_smoke.json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "metrics": metrics}, f, indent=1)
        log(f"smoke test finished in {time.time() - t_start:.0f} s (nothing saved to results/); "
            f"figures and metrics_smoke.json in {out_dir}")
        return metrics, config, files
    save_run("snr-mismatch-val", a.member, config, metrics, files=files,
             note="SNR-mismatch robustness of LMMSE and the experiment-02 MLPs on validation run 0056 "
                  "(odd 50-snapshot blocks reported; LMMSE loading chosen on even blocks); test run 0057 untouched")
    return metrics, config, files


if __name__ == "__main__":
    main()
