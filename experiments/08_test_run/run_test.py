"""The single evaluation on test run 0057, with every choice frozen in frozen.json beforehand.

Nothing is selected on the evaluated run: the LMMSE loading rho comes from the validation baseline
run, the MLPs are the saved weights of the validation runs (SHA-256 checked), and every reported
block of the run counts.

Rehearsal (do this first, as often as needed): the same code path on validation run 0056.
    python experiments/08_test_run/run_test.py --member <name> --run 56 --odd-only
must reproduce the validation numbers of the frozen runs exactly; the script checks this itself.

The test (once, after the freeze commit):
    python experiments/08_test_run/run_test.py --member <name> --run 57 --confirm-test
refuses unless the working tree is clean (the record must point at the frozen commit) and no
earlier test-run record exists in results/.
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
from dataset import DEFAULT_NPZ, TEST_RUN, bootstrap_ci, load_test  # noqa: E402
from estimators import (LMMSE, add_noise, channel_stats, linear_matrix, nmse_db,  # noqa: E402
                        pilot_positions, spline_matrix, withheld)
from neural import load_predictor  # noqa: E402
from runlog import _git, code_state, load_artifacts, save_run  # noqa: E402

FROZEN = os.path.join(HERE, "frozen.json")


def load_frozen(path=FROZEN):
    with open(path, encoding="utf-8") as f:
        fz = json.load(f)
    base = json.load(open(os.path.join(ROOT, fz["main"]["baselines_run"], "run.json"), encoding="utf-8"))
    seeds = [load_artifacts(r) for r in fz["main"]["mlp_runs"]]          # checks every SHA-256
    return fz, base, seeds


def per_block(e, h, gid, nb):
    return np.bincount(gid, np.sum(np.abs(e - h) ** 2, 1), nb)


def evaluate_main(fz, base, seeds, d, mask, log):
    """NMSE of every frozen main-line method on the vectors `mask` of d.Ht (paired block bootstrap)."""
    m = fz["main"]
    Ht = d.Ht[mask]
    _, gid = np.unique(d.block_of[mask], return_inverse=True)
    nb = int(gid.max()) + 1
    boots = np.random.default_rng(0).integers(0, nb, (2000, nb))
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
    log(f"rehearsal: max |difference| to the frozen validation records = {worst:.4f} dB")
    if worst > 1.5e-3:                                                    # 3-decimal rounding
        raise RuntimeError("rehearsal does not reproduce the validation numbers; do not run the test")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--data", default=DEFAULT_NPZ)
    ap.add_argument("--run", type=int, required=True, help="56 = rehearsal on validation, 57 = THE test")
    ap.add_argument("--odd-only", action="store_true", help="rehearsal: report only odd blocks of run 56")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()

    test = a.run == TEST_RUN
    if test:
        if not a.confirm_test or a.odd_only:
            sys.exit("the test run needs --confirm-test (and no --odd-only)")
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

    fz, base, seeds = load_frozen()
    d = load_test(a.data, test_run=a.run, confirm=test)
    mask = d.odd if a.odd_only else np.ones(len(d.Ht), bool)
    log(f"run {a.run:04d}: fit {len(d.Hf)}, evaluated {int(mask.sum())} vectors in "
        f"{len(np.unique(d.block_of[mask]))} blocks; screened fraction {d.screened_fraction}")
    metrics = {"main": evaluate_main(fz, base, seeds, d, mask, log)}
    config = {"run": a.run, "odd_only": a.odd_only, "frozen": fz, "n_vectors": int(mask.sum()),
              "n_blocks": int(len(np.unique(d.block_of[mask]))), "screened_fraction": d.screened_fraction,
              "runtime_s": round(time.time() - t0, 1)}
    if not test:
        metrics["rehearsal_max_abs_diff_db"] = check_rehearsal(metrics["main"], fz, base, seeds, log)
        name = "rehearsal-0056" + ("-odd" if a.odd_only else "")
    else:
        name = f"test-{TEST_RUN:04d}"
    save_run(name, a.member, config, metrics,
             note=("THE single evaluation on test run 0057 with frozen choices" if test
                   else "rehearsal of the test code path on validation run 0056"))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
