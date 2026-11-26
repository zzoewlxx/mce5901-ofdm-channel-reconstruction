"""The single evaluation on test run 0057, with every choice frozen in frozen.json beforehand.

Nothing is selected on the evaluated run, and every reported block of the run counts.

Main line (one antenna, 64 subcarriers, 16 / 8 / 4 pilots): the LMMSE loading rho comes from the
validation baseline run, the MLPs are the saved weights of the validation runs (SHA-256 checked).

SNR mismatch (experiment 04's grid, true x assumed SNR): every design is already frozen per assumed SNR
(LMMSE with the validation loading for that SNR, the SNR-aware MLPs given the assumed value), so the grid
needs no choice on the evaluated run; it is summarised with experiment 04's own summarize().

Joint antenna x subcarrier setting ("spacefreq" of experiment 03: 32 antennas x 16 subcarriers, pilot
patterns 32ant x 4sc and 8ant x 4sc):
  * joint LMMSE with the loading rho chosen on validation (experiment 07 records)
  * GMM: for every seed, the (K, alpha) that seed chose on validation (experiment 06 record), refitted on
    the training runs; a fit is deterministic, so every refit must reproduce the model_sha256 recorded by
    experiment 06 or the script stops
  * jmlp / jresidual: the saved weights of experiment 07 (SHA-256 checked)
  * besides the gains over LMMSE: paired block-bootstrap CIs of GMM - jresidual and GMM - jmlp, which the
    validation runs could not give (06 and 07 were separate runs)

Rehearsal (do this first, as often as needed): the same code path on validation run 0056.
    python experiments/08_test_run/run_test.py --member <name> --run 56 --odd-only [--parts main joint]
must reproduce the validation numbers of the frozen runs exactly; the script checks this itself.

The test (once, after the freeze commit; always both parts):
    python experiments/08_test_run/run_test.py --member <name> --run 57 --confirm-test
refuses unless the working tree is clean (the record must point at the frozen commit) and no
earlier test-run record exists in results/.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # default thread count is far slower on small matrices
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")      # as experiment 06 (silences a harmless sklearn warning)
import sys  # noqa: E402
sys.dont_write_bytecode = True                         # importing other experiments must not write into them
import argparse  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
for _exp in ("03_gmm_feasibility", "04_snr_mismatch", "06_gmm_seeds", "07_joint_mlp"):
    sys.path.insert(0, os.path.join(ROOT, "experiments", _exp))
from dataset import DEFAULT_NPZ, TEST_RUN, bootstrap_ci, load_test  # noqa: E402
from estimators import (LMMSE, add_noise, channel_stats, linear_matrix, nmse_db,  # noqa: E402
                        pilot_positions, spline_matrix, withheld)
from neural import VARIANTS, load_predictor  # noqa: E402
from runlog import _git, code_state, load_artifacts, save_run  # noqa: E402
import gmm_feasibility as g03  # noqa: E402  (cov, mix_estimate)
import run_mismatch as g04  # noqa: E402     (summarize: experiment 04's summary of the SNR grid)
import run_gmm_seeds as g06  # noqa: E402    (kmeans_labels, fit_em: the EM of experiment 06)
import run_joint_mlp as j07  # noqa: E402    (global_cov, pattern_dict, load_joint_predictor)

FROZEN = os.path.join(HERE, "frozen.json")
PARTS = ("main", "mismatch", "joint")
B_TEST = 2000                                          # bootstrap resamples of the evaluated blocks
TOL_DB = 1.5e-3                                        # rehearsal tolerance: 3-decimal rounding of the records


def read_json(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def load_frozen(path=FROZEN):
    with open(path, encoding="utf-8") as f:
        fz = json.load(f)
    base = read_json(os.path.join(fz["main"]["baselines_run"], "run.json"))
    seeds = [load_artifacts(r) for r in fz["main"]["mlp_runs"]]          # checks every SHA-256
    return fz, base, seeds


def load_frozen_joint(fz):
    """Experiment 06 record (GMM configs, model hashes) and the experiment 07 records + weights per pattern."""
    j = fz["joint"]
    rec06 = read_json(os.path.join(j["gmm_run"], "run.json"))
    mlp = {p: load_artifacts(r) for p, r in j["mlp_runs"].items()}     # checks every SHA-256
    return rec06, mlp


def per_block(e, h, gid, nb):
    return np.bincount(gid, np.sum(np.abs(e - h) ** 2, 1), nb)


def ci95(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def evaluate_main(fz, base, seeds, d, mask, log):
    """NMSE of every frozen main-line method on the vectors `mask` of d.Ht (paired block bootstrap)."""
    m = fz["main"]
    Ht = d.Ht[mask]
    _, gid = np.unique(d.block_of[mask], return_inverse=True)
    nb = int(gid.max()) + 1
    boots = np.random.default_rng(0).integers(0, nb, (B_TEST, nb))
    mu, C = channel_stats(d.Hf)
    out = {}
    for n_obs in m["densities"]:
        p, miss = pilot_positions(d.n, n_obs), withheld(d.n, pilot_positions(d.n, n_obs))
        lm = LMMSE(mu, C, p)
        lm.apply_per_sample(Ht[:1, p], 1.0)                               # eigen-decomposition, built once
        Wl, Ws = linear_matrix(d.n, p), spline_matrix(d.n, p)
        models = {}
        for s, (rec, paths) in enumerate(seeds):
            for v in m["variants"]:
                pred, meta = load_predictor(paths[f"{n_obs}_{v}.npz"], lm, miss)
                if abs(meta["scale"] - d.scale) > 1e-12 * d.scale:
                    raise ValueError(f"seed {s} {v}: trained with scale {meta['scale']}, data scale {d.scale}")
                models[(v, s)] = pred
        out[str(n_obs)] = {}
        for snr in m["snrs"]:
            nv = 10 ** (-snr / 10)
            Y = add_noise(d.Ht[:, p], nv, seed=1000 * n_obs + snr)[mask]   # same draw as experiments 01/02
            rho = base["metrics"]["nmse"][str(n_obs)][str(snr)]["lmmse_rho"]
            est = {"linear": (Y @ Wl.T)[:, miss], "spline": (Y @ Ws.T)[:, miss],
                   "lmmse": lm.apply(Y, lm.matrix(nv, rho))[:, miss]}
            for (v, s), pred in models.items():
                est[f"{v}@seed{s}"] = pred(Y, np.full(len(Y), snr))
            H = Ht[:, miss]
            ref_b = np.bincount(gid, np.sum(np.abs(H) ** 2, 1), nb)
            err_b = {k: per_block(e, H, gid, nb) for k, e in est.items()}
            for v in m["variants"]:                                       # seed ensemble: per-block errors averaged
                err_b[f"{v}@mean"] = np.mean([err_b[f"{v}@seed{s}"] for s in range(len(seeds))], axis=0)
            row, bs = {"lmmse_rho": rho}, {}
            for k, eb in err_b.items():
                bs[k], ci = bootstrap_ci(eb, ref_b, boots)
                point = nmse_db(est[k], H) if k in est else float(10 * np.log10(eb.sum() / ref_b.sum()))
                row[k] = {"nmse_db": round(point, 3), "ci95": ci}
            for k in [x for x in err_b if x != "lmmse"]:                  # positive = better than LMMSE
                g = bs["lmmse"] - bs[k]
                row[f"{k}_gain_over_lmmse_db"] = {"value": round(row["lmmse"]["nmse_db"] - row[k]["nmse_db"], 3),
                                                  "ci95": [round(float(np.percentile(g, 2.5)), 3),
                                                           round(float(np.percentile(g, 97.5)), 3)]}
            out[str(n_obs)][str(snr)] = row
            log(f"  {n_obs:2d} pilots {snr:2d} dB: linear {row['linear']['nmse_db']:6.2f} lmmse {row['lmmse']['nmse_db']:6.2f} "
                + " ".join(f"{v} {row[f'{v}@mean']['nmse_db']:6.2f}" for v in m["variants"]))
    return out


def check_rehearsal(out, fz, base, seeds, log):
    """On run 56 odd blocks the numbers must equal the frozen validation records (to rounding)."""
    worst = 0.0
    for dens, rows in out.items():
        for snr, row in rows.items():
            ref = base["metrics"]["nmse"][dens][snr]
            pairs = [(row[k]["nmse_db"], ref[k]["nmse_db"]) for k in ("linear", "spline", "lmmse")]
            for s, (rec, _) in enumerate(seeds):
                vr = rec["metrics"]["nmse"][dens][snr]
                pairs += [(row[f"{v}@seed{s}"]["nmse_db"], vr[v]["nmse_db"]) for v in fz["main"]["variants"]]
            worst = max(worst, max(abs(a - b) for a, b in pairs))
    log(f"rehearsal (main): max |difference| to the frozen validation records = {worst:.4f} dB")
    if worst > TOL_DB:
        raise RuntimeError("rehearsal does not reproduce the validation numbers; do not run the test")
    return worst


# ----------------------------------------------------------------------------- SNR mismatch
def evaluate_mismatch(fz, base, seeds, d, mask, log):
    """Experiment 04's (true, assumed) grid on the vectors `mask` of d.Ht with the frozen designs.

    Same arithmetic as run_mismatch.evaluate_density, except that the LMMSE loading for an assumed SNR a
    is read from the frozen baselines record instead of being chosen (04 chose it the same way, on the even
    validation blocks under matched noise; check_rehearsal_mismatch confirms both agree)."""
    mm = fz["mismatch"]
    snrs = list(mm["snrs"])
    Ht = d.Ht[mask]
    _, gid = np.unique(d.block_of[mask], return_inverse=True)
    nb = int(gid.max()) + 1
    boots = np.random.default_rng(0).integers(0, nb, (B_TEST, nb))
    mu, C = channel_stats(d.Hf)
    seed_ids = [rec["config"]["seed"] for rec, _ in seeds]
    out = {}
    for n_obs in mm["densities"]:
        t0 = time.time()
        p = pilot_positions(d.n, n_obs)
        miss = withheld(d.n, p)
        lm = LMMSE(mu, C, p)                                  # one object: the residuals share its eigen cache
        Wl = linear_matrix(d.n, p)
        Hr = Ht[:, miss]
        ref_tot = np.sum(np.abs(Hr) ** 2)
        S = len(snrs)
        rho = {s: base["metrics"]["nmse"][str(n_obs)][str(s)]["lmmse_rho"] for s in snrs}
        G = {s: lm.matrix(10 ** (-s / 10), rho[s]) for s in snrs}
        nets = {}
        for v in mm["variants"]:
            aware, residual = VARIANTS[v][1], VARIANTS[v][2]
            if residual and not aware:
                raise ValueError(f"{v}: a residual variant without SNR input is not supported")
            preds = [load_predictor(paths[f"{n_obs}_{v}.npz"], lm, miss)[0] for _, paths in seeds]
            shape = (len(seeds), S, S) if aware else (len(seeds), S)
            nets[v] = {"aware": aware, "seeds": seed_ids, "preds": preds,
                       "nmse": np.zeros(shape), "err": np.zeros(shape + (nb,))}
        raw = {"n_obs": n_obs, "pilots": p.tolist(), "rho": rho,
               "ref_b": np.bincount(gid, np.sum(np.abs(Hr) ** 2, 1), nb),
               "linear": {"nmse": np.zeros(S), "err": np.zeros((S, nb))},
               "lmmse": {"nmse": np.zeros((S, S)), "err": np.zeros((S, S, nb))}, "nets": nets}

        def put(slot, idx, est):
            sq = np.abs(est - Hr) ** 2
            slot["nmse"][idx] = 10 * np.log10(sq.sum() / ref_tot)
            slot["err"][idx] = np.bincount(gid, sq.sum(1), nb)

        for i, t in enumerate(snrs):
            Y = add_noise(d.Ht[:, p], 10 ** (-t / 10), seed=1000 * n_obs + t)[mask]   # same draw as 01 / 02 / 04
            put(raw["linear"], i, (Y @ Wl.T)[:, miss])
            for net in nets.values():
                if not net["aware"]:
                    for k, pred in enumerate(net["preds"]):
                        put(net, (k, i), pred(Y, np.zeros(len(Y))))       # blind: the SNR input is ignored
            for j, s in enumerate(snrs):
                put(raw["lmmse"], (i, j), lm.apply(Y, G[s])[:, miss])
                s_vec = np.full(len(Y), float(s))
                for net in nets.values():
                    if net["aware"]:
                        for k, pred in enumerate(net["preds"]):
                            put(net, (k, i, j), pred(Y, s_vec))
        for net in nets.values():
            del net["preds"]
        out[str(n_obs)] = g04.summarize(raw, snrs, boots)
        wc = out[str(n_obs)]["worst_case_loss_db"]
        msg = (f"  mismatch {n_obs:2d} pilots ({time.time() - t0:.0f} s): worst loss vs own matched SNR: "
               + "  ".join(f"{v} {wc[v]['vs_own_matched']['loss_db']:.2f}" for v in ["lmmse"] + list(nets)
                           if wc[v]["vs_own_matched"]))
        if "mlp_blind" in nets:
            msg += f"; blind MLP vs matched LMMSE at most {wc['mlp_blind']['vs_matched_lmmse']['loss_db']:.2f}"
        log(msg)
    return out


def check_rehearsal_mismatch(out, fz, log):
    """On run 56 odd blocks the grid must equal experiment 04's record: LMMSE loadings, the LMMSE and linear
    grids, and every seed's NMSE grid of every variant."""
    rec = read_json(os.path.join(fz["mismatch"]["validation_run"], "run.json"))["metrics"]["grid"]
    worst, where, n = 0.0, "", 0
    for dens, summ in out.items():
        g, r = summ["grid"], rec[dens]
        if {k: float(v) for k, v in g["lmmse_rho"].items()} != {k: float(v) for k, v in r["lmmse_rho"].items()}:
            raise RuntimeError(f"{dens} pilots: frozen LMMSE loadings differ from experiment 04's choice")
        pairs = [("lmmse", g["lmmse"]["nmse_db"], r["lmmse"]["nmse_db"]),
                 ("linear", g["linear"]["nmse_db"], r["linear"]["nmse_db"])]
        for v in fz["mismatch"]["variants"]:
            for sd, grid in g[v]["per_seed_nmse_db"].items():
                pairs.append((f"{v} seed {sd}", grid, r[v]["per_seed_nmse_db"][sd]))
        for name, a, b in pairs:
            diff = np.abs(np.asarray(a, float) - np.asarray(b, float))
            n += diff.size
            if diff.max() > worst:
                worst, where = float(diff.max()), f"{dens} pilots {name}"
    log(f"rehearsal (mismatch): max |difference| to experiment 04's record = {worst:.4f} dB over {n} numbers"
        + (f" ({where})" if where else ""))
    if worst > TOL_DB:
        raise RuntimeError("mismatch rehearsal does not reproduce experiment 04; do not run the test")
    return worst


# ----------------------------------------------------------------------------- joint setting
def load_spacefreq(path, run, confirm=False, thr=-30.0, block_len=50):
    """Experiment 03's "spacefreq" vectors (gmm_feasibility.load, same arithmetic) with `run` as the
    evaluated run: fit on runs 0053-0055, snapshots whose 32 antennas all pass the screening, 16 of the 64
    subcarriers (entry = antenna * 16 + grid subcarrier), one global scale from the fit vectors."""
    if run == TEST_RUN and not confirm:
        raise PermissionError(f"run {TEST_RUN:04d} is the held-out test run; it needs confirm=True")
    z = np.load(path)
    csi, q, runs, index = z["csi"], z["quality"], z["run"], z["index"]
    good = (q <= thr).all(1)
    sc = pilot_positions(64, 16)

    def vec(m):
        idx = np.flatnonzero(m & good)
        return csi[idx][:, :, sc].reshape(len(idx), -1).astype(np.complex128), idx
    Xf, _ = vec(np.isin(runs, (53, 54, 55)))
    Xe, se = vec(runs == run)
    scale = 1 / np.sqrt(np.mean(np.abs(Xf) ** 2))
    n_run = int(np.sum(runs == run))
    return dict(Xf=Xf * scale, Xe=Xe * scale, block=index[se] // block_len, scale=float(scale),
                n_run_snapshots=n_run, screened_fraction=round(1 - len(se) / n_run, 4))


def evaluate_joint(fz, rec06, mlp, J, mask, log):
    """NMSE of joint LMMSE, GMM (refitted, hash-checked) and the joint MLPs on the vectors `mask` of J["Xe"]."""
    j = fz["joint"]
    Xf, H = J["Xf"], J["Xe"][mask]
    n, d = Xf.shape
    _, gid = np.unique(J["block"][mask], return_inverse=True)
    nb = int(gid.max()) + 1
    boots = np.random.default_rng(0).integers(0, nb, (B_TEST, nb))
    cfg06 = rec06["config"]
    if abs(rec06["metrics"]["data"]["scale"] - J["scale"]) > 1e-12 * J["scale"] or rec06["metrics"]["data"]["n_fit"] != n:
        raise ValueError("fit data differ from experiment 06's")
    if cfg06["em_max"] < n or cfg06["gmm_rho"] != 0.0:
        raise ValueError("experiment 06 used an EM subsample or a GMM loading; this script assumes neither")
    for p, (rec, _) in mlp.items():
        if abs(rec["config"]["scale"] - J["scale"]) > 1e-12 * J["scale"] or rec["config"]["n_fit"] != n:
            raise ValueError(f"{p}: fit data differ from experiment 07's")
    pats = j07.pattern_dict()
    Cg = j07.global_cov(Xf)                    # LMMSE and residual base: cov(all fit vectors) + 1e-6 loading (03)
    Cg_fit = g03.cov(Xf)                       # GMM: k-means phase reference and shrinkage target (06)

    rows = {(p, s) for p in j["patterns"] for s in j["snrs"]}
    Y = {(p, s): add_noise(J["Xe"][:, pats[p]], 10 ** (-s / 10), seed=7919 * len(pats[p]) + s)[mask]
         for p, s in rows}                     # same draw as experiments 03 / 06 / 07
    M = {p: np.setdiff1d(np.arange(d), pats[p]) for p in j["patterns"]}
    est = {r: {} for r in rows}                # (pattern, snr) -> {method: estimate of the withheld entries}

    for p in j["patterns"]:                    # joint LMMSE and the joint MLPs
        P = pats[p]
        lm = LMMSE(np.zeros(d, complex), Cg, P)
        rec07, paths = mlp[p]
        for s in j["snrs"]:
            rho = rec07["metrics"]["nmse"][p][str(s)]["lmmse"]["rho"]
            if rho != rec06["metrics"]["lmmse"][p][str(s)]["rho"]:
                raise ValueError(f"{p} {s} dB: experiments 06 and 07 chose different LMMSE loadings")
            est[p, s]["lmmse"] = g03.mix_estimate(Y[p, s], P, np.ones(1), Cg[None], 10 ** (-s / 10), rho)[:, M[p]]
        for v in j["mlp_variants"]:
            for seed in j["mlp_seeds"]:
                pred, meta = j07.load_joint_predictor(paths[f"{p.replace(' x ', '-')}_{v}_seed{seed}.npz"], lm)
                if abs(meta["scale"] - J["scale"]) > 1e-12 * J["scale"]:
                    raise ValueError(f"{p} {v} seed {seed}: trained with scale {meta['scale']}, data {J['scale']}")
                for s in j["snrs"]:
                    est[p, s][f"{v}@seed{seed}"] = pred(Y[p, s], s)
        log(f"  joint {p}: LMMSE and {len(j['mlp_variants']) * len(j['mlp_seeds'])} MLPs evaluated")

    hashes = {}
    for seed in j["gmm_seeds"]:                # GMM: each seed's own validation choice per row, refitted
        rows06 = rec06["metrics"]["per_seed"][str(seed)]["rows"]
        need = sorted({rows06[p][str(s)]["chosen"] for p, s in rows})
        for name in need:
            K, alpha = next((r["K"], r["alpha"]) for pr in rows06.values() for r in pr.values() if r["chosen"] == name)
            t = time.time()
            labels = g06.kmeans_labels(Xf, Cg_fit, K, seed)
            pi, C, info = g06.fit_em(Xf, K, alpha, labels, Cg_fit, iters=cfg06["em_iters_cap"], tol=cfg06["em_tol"])
            want = rec06["metrics"]["per_seed"][str(seed)]["fits"][name]["model_sha256"]
            hashes[f"seed{seed} {name}"] = {"refit": info["model_sha256"], "experiment06": want,
                                            "iters": info["iters"]}
            if info["model_sha256"] != want:
                raise RuntimeError(f"GMM seed {seed} {name}: refit hash {info['model_sha256']} differs from "
                                   f"experiment 06's {want}; the frozen model is not reproduced")
            for p, s in sorted(rows):
                if rows06[p][str(s)]["chosen"] == name:
                    est[p, s][f"gmm@seed{seed}"] = g03.mix_estimate(Y[p, s], pats[p], pi, C, 10 ** (-s / 10))[:, M[p]]
            del C
            log(f"  GMM seed {seed} {name}: refit in {time.time() - t:.0f} s ({info['iters']} EM iterations), "
                f"hash {info['model_sha256']} = experiment 06")

    families = {"gmm": j["gmm_seeds"], **{v: j["mlp_seeds"] for v in j["mlp_variants"]}}
    out = {}
    for p in j["patterns"]:
        out[p] = {}
        Hm = H[:, M[p]]
        ref_b = np.bincount(gid, np.sum(np.abs(Hm) ** 2, 1), nb)
        for s in j["snrs"]:
            e = est[p, s]
            err_b = {k: per_block(x, Hm, gid, nb) for k, x in e.items()}
            for f, sds in families.items():                              # seed ensembles
                err_b[f"{f}@mean"] = np.mean([err_b[f"{f}@seed{sd}"] for sd in sds], axis=0)
            row, bs = {"lmmse_rho": rec07_rho(mlp, p, s)}, {}
            for k, eb in err_b.items():
                bs[k], ci = bootstrap_ci(eb, ref_b, boots)
                point = nmse_db(e[k], Hm) if k in e else float(10 * np.log10(eb.sum() / ref_b.sum()))
                row[k] = {"nmse_db": round(point, 3), "ci95": ci}
            for k in [x for x in err_b if x != "lmmse"]:                  # positive = better than LMMSE
                row[f"{k}_gain_over_lmmse_db"] = {"value": round(row["lmmse"]["nmse_db"] - row[k]["nmse_db"], 3),
                                                  "ci95": ci95(bs["lmmse"] - bs[k])}
            for v in j["mlp_variants"]:                                   # positive = GMM better
                row[f"gmm@mean_minus_{v}@mean_db"] = {
                    "value": round(row[f"{v}@mean"]["nmse_db"] - row["gmm@mean"]["nmse_db"], 3),
                    "ci95": ci95(bs[f"{v}@mean"] - bs["gmm@mean"])}
            out[p][str(s)] = row
            msg = (f"  {p:>12s} {s:2d} dB: LMMSE {row['lmmse']['nmse_db']:7.2f} | gain "
                   + " ".join(f"{f} {row[f'{f}@mean_gain_over_lmmse_db']['value']:+.2f}" for f in families))
            for v in j["mlp_variants"]:
                dv = row[f"gmm@mean_minus_{v}@mean_db"]
                msg += f" | GMM - {v} {dv['value']:+.2f} [{dv['ci95'][0]:+.2f}, {dv['ci95'][1]:+.2f}]"
            log(msg)
    return out, hashes


def rec07_rho(mlp, p, s):
    return mlp[p][0]["metrics"]["nmse"][p][str(s)]["lmmse"]["rho"]


def check_rehearsal_joint(out, fz, rec06, mlp, log):
    """On run 56 odd blocks every joint number must equal the experiment 06 / 07 records (to rounding)."""
    j = fz["joint"]
    pairs = []
    for p, rows in out.items():
        for s, row in rows.items():
            pairs.append((row["lmmse"]["nmse_db"], rec06["metrics"]["lmmse"][p][s]["nmse_db"], f"{p} {s} lmmse"))
            for seed in j["gmm_seeds"]:
                ref = rec06["metrics"]["per_seed"][str(seed)]["rows"][p][s]["nmse_db"]
                pairs.append((row[f"gmm@seed{seed}"]["nmse_db"], ref, f"{p} {s} gmm seed {seed}"))
            r07 = mlp[p][0]["metrics"]["nmse"][p][s]
            for v in j["mlp_variants"]:
                for seed in j["mlp_seeds"]:
                    pairs.append((row[f"{v}@seed{seed}"]["nmse_db"], r07[v]["seeds"][str(seed)]["nmse_db"],
                                  f"{p} {s} {v} seed {seed}"))
    worst = max(pairs, key=lambda x: abs(x[0] - x[1]))
    diff = abs(worst[0] - worst[1])
    log(f"rehearsal (joint): max |difference| to the experiment 06 / 07 records = {diff:.4f} dB "
        f"over {len(pairs)} numbers ({worst[2]})")
    if diff > TOL_DB:
        raise RuntimeError("joint rehearsal does not reproduce the validation numbers; do not run the test")
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--data", default=DEFAULT_NPZ)
    ap.add_argument("--run", type=int, required=True, help="56 = rehearsal on validation, 57 = THE test")
    ap.add_argument("--odd-only", action="store_true", help="rehearsal: report only odd blocks of run 56")
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS,
                    help="rehearsal only: evaluate a subset (the test always evaluates every part)")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    parts = [p for p in PARTS if p in a.parts]

    test = a.run == TEST_RUN
    if test:
        if not a.confirm_test or a.odd_only or parts != list(PARTS):
            sys.exit("the test run needs --confirm-test, no --odd-only and every part")
        uncommitted, new_code = code_state()
        if uncommitted is not False:
            sys.exit(f"refusing: commit the frozen code and frozen.json first (untracked code: {new_code})")
        rel = os.path.relpath(FROZEN, ROOT).replace(os.sep, "/")
        if _git("ls-files", rel) != rel or _git("status", "--porcelain", "--", rel):
            sys.exit(f"refusing: {rel} must be committed and unmodified")
        if json.load(open(FROZEN, encoding="utf-8")).get("status") != "FROZEN":
            sys.exit("refusing: frozen.json status is not FROZEN")
        earlier = glob.glob(os.path.join(ROOT, "results", f"*_test-{TEST_RUN:04d}*"))
        if earlier:
            sys.exit(f"refusing: the test run was already evaluated: {earlier}")
    elif a.run != 56:
        sys.exit("only run 56 (rehearsal) or 57 (test) are meaningful here")

    fz, base, seeds = load_frozen()                                       # SHA-256 checks before any data
    if "joint" in parts:
        rec06, mlp = load_frozen_joint(fz)
    config = {"run": a.run, "odd_only": a.odd_only, "parts": parts, "frozen": fz, "bootstrap_B": B_TEST}
    metrics = {}
    name = f"test-{TEST_RUN:04d}" if test else (
        "rehearsal-0056" + ("-odd" if a.odd_only else "") + ("" if parts == list(PARTS) else "-" + "-".join(parts)))
    partial = os.path.join(ROOT, "data", "logs", f"{time.strftime('%Y%m%d-%H%M')}_{name}_partial.json")

    def keep():   # a crash in a later part must not lose the numbers already computed (data/ is not in git)
        os.makedirs(os.path.dirname(partial), exist_ok=True)
        with open(partial, "w", encoding="utf-8") as f:
            json.dump({"config": config, "metrics": metrics}, f, indent=1)

    if "main" in parts or "mismatch" in parts:
        d = load_test(a.data, test_run=a.run, confirm=test)
        mask = d.odd if a.odd_only else np.ones(len(d.Ht), bool)
        nb = len(np.unique(d.block_of[mask]))
        log(f"[main] run {a.run:04d}: fit {len(d.Hf)}, evaluated {int(mask.sum())} vectors in {nb} blocks; "
            f"screened fraction {d.screened_fraction}")
        config["main_data"] = {"n_vectors": int(mask.sum()), "n_blocks": int(nb),
                               "screened_fraction": d.screened_fraction}
        if "main" in parts:
            metrics["main"] = evaluate_main(fz, base, seeds, d, mask, log)
            keep()
            if not test:
                metrics["rehearsal_max_abs_diff_db"] = check_rehearsal(metrics["main"], fz, base, seeds, log)
        if "mismatch" in parts:
            t = time.time()
            log("[mismatch] true x assumed SNR grid with the frozen designs")
            metrics["mismatch"] = evaluate_mismatch(fz, base, seeds, d, mask, log)
            config["mismatch_runtime_s"] = round(time.time() - t, 1)
            keep()
            if not test:
                metrics["rehearsal_mismatch_max_abs_diff_db"] = check_rehearsal_mismatch(metrics["mismatch"], fz, log)
        del d, mask
        gc.collect()
    if "joint" in parts:
        t = time.time()
        J = load_spacefreq(a.data, a.run, confirm=test)
        mask = (J["block"] % 2 == 1) if a.odd_only else np.ones(len(J["Xe"]), bool)
        nb = len(np.unique(J["block"][mask]))
        log(f"[joint] run {a.run:04d}: fit {J['Xf'].shape}, evaluated {int(mask.sum())} snapshots in {nb} blocks; "
            f"{J['screened_fraction']:.2%} of the run's snapshots screened out (not all 32 antennas ok)")
        metrics["joint"], metrics["joint_gmm_refit_hashes"] = evaluate_joint(fz, rec06, mlp, J, mask, log)
        config["joint_data"] = {"n_vectors": int(mask.sum()), "n_blocks": int(nb), "dim": int(J["Xe"].shape[1]),
                                "n_run_snapshots": J["n_run_snapshots"], "screened_fraction": J["screened_fraction"],
                                "runtime_s": round(time.time() - t, 1)}
        keep()
        if not test:
            metrics["rehearsal_joint_max_abs_diff_db"] = check_rehearsal_joint(metrics["joint"], fz, rec06, mlp, log)
    config["runtime_s"] = round(time.time() - t0, 1)
    save_run(name, a.member, config, metrics,
             note=("THE single evaluation on test run 0057 with frozen choices" if test
                   else "rehearsal of the test code path on validation run 0056"))
    if os.path.exists(partial):
        os.remove(partial)                                  # the full record is saved
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
