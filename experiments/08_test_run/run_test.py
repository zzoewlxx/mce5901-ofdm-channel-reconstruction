"""The single evaluation on test run 0057, with every choice frozen in frozen.json beforehand.

Nothing is selected on the evaluated run, and every reported block of the run counts.

Parts, evaluated in this order (each one reduces every estimate to per-block squared errors at once):
  joint     "spacefreq" of experiment 03 (32 antennas x 16 subcarriers; pilots 32ant x 4sc and 8ant x 4sc):
            joint LMMSE with the validation loadings of experiment 07; the GMM of experiment 06, refitted
            per seed with the (K, alpha) that seed chose on validation; the joint MLPs of experiment 07
            (saved weights, SHA-256 checked). Besides the gains over LMMSE: paired CIs of the GMM's gain
            over each joint MLP, which the separate validation runs could not give.
  main      one antenna, 64 subcarriers, 16 / 8 / 4 pilots: linear, spline, LMMSE (validation loading of
            the baselines record) and the MLP variants of experiment 02 (saved weights, SHA-256 checked).
  mismatch  experiment 04's true x assumed SNR grid. Every design is frozen per assumed SNR (LMMSE with the
            validation loading for that SNR, the SNR-aware MLPs given the assumed value), so the grid needs
            no choice on the evaluated run; it is summarised with experiment 04's own summarize().

Pre-flight, before ANY vector of the evaluated run is extracted: the BLAS thread count and library versions;
the SHA-256 of every weight file; the consistency of the 06 / 07 records with the fit data; and the six GMM
refits, which use the training runs only and must reproduce experiment 06's k-means labelling and model
hash bit for bit (k-means is retried on the training data if a multi-threaded reduction changed a label).

Rehearsal (as often as needed; reproduces every validation record it can, and must pass before the test):
    python experiments/08_test_run/run_test.py --member <name> --run 56 --odd-only [--parts ...]
Full-size dry run on run 56 (all blocks, like the test; no reproduction check is possible there):
    python experiments/08_test_run/run_test.py --member <name> --run 56
The test (once; all parts):
    python experiments/08_test_run/run_test.py --member <name> --run 57 --confirm-test
It refuses unless: no Python code under src/, experiments/, tools/ is uncommitted or untracked; frozen.json
and every record it names are committed and unmodified, and frozen.json has status FROZEN; a COMMITTED,
passing three-part odd-block rehearsal exists that started on clean code which did not change while it ran,
and whose code (*.py), frozen records, frozen choices (frozen.json's main / mismatch / joint sections), data
file (SHA-256) and environment equal the current ones; no completed test record exists; and no earlier
attempt exists (results/*test-0057*, data/logs/*test-0057*) unless --after-failed-attempt "<reason>" is
given, in which case the earlier attempt must have used the same code, frozen choices and data. After the
pre-flight and before touching run 57 it writes results/<stamp>_<member>_test-0057-started/started.json
(and crashed.json there if it fails); commit that folder and the final record. Documentation (README.md,
report) may change between rehearsal and test. Do not commit code while the test runs: the record stores the
start and end state.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"   # every frozen record used 4; GMM refits are bit-exact only at 4
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")      # as experiment 06 (silences a harmless sklearn warning)
import sys  # noqa: E402
sys.dont_write_bytecode = True                         # importing other experiments must not write into them
import argparse  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
for _exp in ("03_gmm_feasibility", "04_snr_mismatch", "06_gmm_seeds", "07_joint_mlp"):
    sys.path.insert(0, os.path.join(ROOT, "experiments", _exp))
from dataset import DEFAULT_NPZ, TEST_RUN, load_test  # noqa: E402
from estimators import (LMMSE, add_noise, channel_stats, linear_matrix,  # noqa: E402
                        pilot_positions, spline_matrix, withheld)
from neural import VARIANTS, load_predictor  # noqa: E402
from runlog import _git, load_artifacts, save_run, sha256  # noqa: E402
import gmm_feasibility as g03  # noqa: E402  (cov, mix_estimate)
import run_mismatch as g04  # noqa: E402     (summarize: experiment 04's summary of the SNR grid)
import run_gmm_seeds as g06  # noqa: E402    (kmeans_labels, fit_em, digest: the EM of experiment 06)
import run_joint_mlp as j07  # noqa: E402    (global_cov, pattern_dict, load_joint_predictor)

FROZEN = os.path.join(HERE, "frozen.json")
FROZEN_REL = "experiments/08_test_run/frozen.json"
PARTS = ("joint", "main", "mismatch")                  # evaluation order
B_TEST = 2000                                          # bootstrap resamples of the evaluated blocks
MERGE = 4                                              # robustness CI: 4 neighbouring 50-snapshot blocks merged
TOL_DB = 1.5e-3                                        # rehearsal tolerance: 3-decimal rounding of the records
KMEANS_TRIES = 3                                       # k-means retries (training data only) per GMM refit
TEST_TAG = f"test-{TEST_RUN:04d}"

DEFINITIONS = {
    "nmse_db": "10 log10(sum of squared errors / sum of squared entries) over the withheld entries of all "
               "evaluated vectors; 3 decimals (as the validation records, so the rehearsal can compare)",
    "ci95": "paired block bootstrap over the evaluated 50-snapshot blocks, B=2000, numpy default_rng(0); the "
            "2.5 / 97.5 percentiles; 4 decimals",
    "ci95_200snap": "the same with the evaluated blocks of each 200-snapshot window of the run merged (block "
                    "index // 4; all 4 blocks on the test run, the 2 odd ones in an odd-block rehearsal; "
                    "default_rng(1)): a robustness check of the CI against correlation between neighbouring blocks",
    "ci_excludes_zero": "from the UNROUNDED percentiles: +1 if the CI lies above 0, -1 below 0, 0 otherwise",
    "<k>_gain_over_lmmse_db": "NMSE(LMMSE) - NMSE(k) from unrounded NMSEs, 4 decimals; positive = k better",
    "gmm@mean_gain_over_<v>@mean_db": "NMSE(<v>@mean) - NMSE(gmm@mean); positive = the GMM better",
    "<k>@seed<s>": "one model (seed s)",
    "<k>@mean": "seed average: per-block squared errors averaged over the seeds, then NMSE (an average "
                "model, NOT averaged predictions)",
    "mismatch": "experiment 04's summarize() on the true x assumed grid: gains from unrounded NMSEs (3 decimals), "
                "gain CIs to 4 decimals with gain_ci_excludes_zero from the unrounded percentiles; B=2000 here "
                "(04 used 1000)",
}


# ----------------------------------------------------------------------------- helpers
def read_json(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def git_ok(*args):
    """Exit status 0 of a git command (e.g. diff --quiet)."""
    return subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def environment():
    import sklearn
    import threadpoolctl
    blas = [p for p in threadpoolctl.threadpool_info() if p.get("internal_api") == "openblas"]
    return {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "threadpoolctl": threadpoolctl.__version__,
            "openblas_threads": sorted({int(p["num_threads"]) for p in blas}),
            "openblas_version": sorted({str(p.get("version")) for p in blas})}


def ci(v, nd=4):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(lo), nd), round(float(hi), nd)], (1 if lo > 0 else -1 if hi < 0 else 0)


class Blocks:
    """Evaluated 50-snapshot blocks of one part: block index per vector, bootstrap resamples, merged blocks."""

    def __init__(self, block_ids):
        ids, self.gid = np.unique(block_ids, return_inverse=True)
        self.nb = int(self.gid.max()) + 1
        self.boots = np.random.default_rng(0).integers(0, self.nb, (B_TEST, self.nb))
        _, self.group = np.unique(ids // MERGE, return_inverse=True)   # same 200-snapshot window of the run
        self.nb4 = int(self.group.max()) + 1
        self.boots4 = np.random.default_rng(1).integers(0, self.nb4, (B_TEST, self.nb4))

    def err(self, e, h):
        return np.bincount(self.gid, np.sum(np.abs(e - h) ** 2, 1), self.nb)

    def ref(self, h):
        return np.bincount(self.gid, np.sum(np.abs(h) ** 2, 1), self.nb)

    def boot_db(self, eb, rb, merged=False):
        if merged:
            eb, rb, idx = np.bincount(self.group, eb, self.nb4), np.bincount(self.group, rb, self.nb4), self.boots4
        else:
            idx = self.boots
        return 10 * np.log10(eb[idx].sum(1) / rb[idx].sum(1))


def summarize_row(blk, ref_b, err, extra=None):
    """NMSE (+CI) of every method and the paired gain of every method over LMMSE, from per-block errors."""
    pt = {k: float(10 * np.log10(e.sum() / ref_b.sum())) for k, e in err.items()}
    bs = {k: blk.boot_db(e, ref_b) for k, e in err.items()}
    bs4 = {k: blk.boot_db(e, ref_b, merged=True) for k, e in err.items()}
    row = dict(extra or {})
    for k in err:
        row[k] = {"nmse_db": round(pt[k], 3), "ci95": ci(bs[k])[0]}

    def compare(a, b):                                  # NMSE(a) - NMSE(b): positive = b better
        c, z = ci(bs[a] - bs[b])
        return {"value": round(pt[a] - pt[b], 4), "ci95": c, "ci_excludes_zero": z,
                "ci95_200snap": ci(bs4[a] - bs4[b])[0]}
    for k in err:
        if k != "lmmse":
            row[f"{k}_gain_over_lmmse_db"] = compare("lmmse", k)
    return row, compare


def max_abs_diff(pairs, what, log):
    """Largest |a - b| over (name, a, b) pairs of numbers or equal-shape grids; non-finite values are an error."""
    worst, where, n = 0.0, "", 0
    for name, a, b in pairs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise RuntimeError(f"rehearsal ({what}): {name}: shapes {a.shape} / {b.shape} or non-finite values")
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        n += a.size
        if d > worst:
            worst, where = d, name
    log(f"rehearsal ({what}): max |difference| to the validation records = {worst:.4f} dB over {n} numbers"
        + (f" ({where})" if where else ""))
    if worst > TOL_DB:
        raise RuntimeError(f"{what} rehearsal does not reproduce the validation records; do not run the test")
    return worst


# ----------------------------------------------------------------------------- frozen inputs
def load_frozen(path=FROZEN):
    with open(path, encoding="utf-8") as f:
        fz = json.load(f)
    base = read_json(os.path.join(fz["main"]["baselines_run"], "run.json"))
    seeds = [load_artifacts(r) for r in fz["main"]["mlp_runs"]]          # checks every SHA-256
    rec06 = read_json(os.path.join(fz["joint"]["gmm_run"], "run.json"))
    mlp = {p: load_artifacts(r) for p, r in fz["joint"]["mlp_runs"].items()}
    return fz, base, seeds, rec06, mlp


def frozen_records(fz):
    """Every run folder frozen.json names."""
    return ([fz["main"]["baselines_run"], *fz["main"]["mlp_runs"], fz["mismatch"]["validation_run"],
             fz["joint"]["gmm_run"], *fz["joint"]["mlp_runs"].values()])


# ----------------------------------------------------------------------------- main line
def evaluate_main(fz, base, seeds, d, mask, log):
    """Every frozen main-line method on the vectors `mask` of d.Ht."""
    m = fz["main"]
    Ht = d.Ht[mask]
    blk = Blocks(d.block_of[mask])
    mu, C = channel_stats(d.Hf)
    out = {}
    for n_obs in m["densities"]:
        p = pilot_positions(d.n, n_obs)
        miss = withheld(d.n, p)
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
        H = Ht[:, miss]
        ref_b = blk.ref(H)
        out[str(n_obs)] = {}
        for snr in m["snrs"]:
            nv = 10 ** (-snr / 10)
            Y = add_noise(d.Ht[:, p], nv, seed=1000 * n_obs + snr)[mask]   # same draw as experiments 01/02
            rho = base["metrics"]["nmse"][str(n_obs)][str(snr)]["lmmse_rho"]
            err = {"linear": blk.err((Y @ Wl.T)[:, miss], H), "spline": blk.err((Y @ Ws.T)[:, miss], H),
                   "lmmse": blk.err(lm.apply(Y, lm.matrix(nv, rho))[:, miss], H)}
            for (v, s), pred in models.items():
                err[f"{v}@seed{s}"] = blk.err(pred(Y, np.full(len(Y), snr)), H)
            for v in m["variants"]:                                       # seed average: per-block errors
                err[f"{v}@mean"] = np.mean([err[f"{v}@seed{s}"] for s in range(len(seeds))], axis=0)
            row, _ = summarize_row(blk, ref_b, err, {"lmmse_rho": rho})
            out[str(n_obs)][str(snr)] = row
            log(f"  {n_obs:2d} pilots {snr:2d} dB: linear {row['linear']['nmse_db']:6.2f} lmmse {row['lmmse']['nmse_db']:6.2f} "
                + " ".join(f"{v} {row[f'{v}@mean']['nmse_db']:6.2f}" for v in m["variants"]))
    return out, blk.nb


def check_rehearsal_main(out, fz, base, seeds, log):
    """Run 56 odd blocks: every NMSE must equal the frozen validation records (to rounding)."""
    pairs = []
    for dens, rows in out.items():
        for snr, row in rows.items():
            ref = base["metrics"]["nmse"][dens][snr]
            pairs += [(f"{dens}/{snr}/{k}", row[k]["nmse_db"], ref[k]["nmse_db"]) for k in ("linear", "spline", "lmmse")]
            for s, (rec, _) in enumerate(seeds):
                vr = rec["metrics"]["nmse"][dens][snr]
                pairs += [(f"{dens}/{snr}/{v}@seed{s}", row[f"{v}@seed{s}"]["nmse_db"], vr[v]["nmse_db"])
                          for v in fz["main"]["variants"]]
    return max_abs_diff(pairs, "main", log)


# ----------------------------------------------------------------------------- SNR mismatch
def evaluate_mismatch(fz, base, seeds, d, mask, log):
    """Experiment 04's (true, assumed) grid on the vectors `mask` of d.Ht with the frozen designs.

    Same arithmetic as run_mismatch.evaluate_density, except that the LMMSE loading for an assumed SNR a
    is read from the frozen baselines record instead of being chosen (04 chose it the same way, on the even
    validation blocks under matched noise; check_rehearsal_mismatch confirms both agree)."""
    mm = fz["mismatch"]
    snrs = list(mm["snrs"])
    Ht = d.Ht[mask]
    blk = Blocks(d.block_of[mask])
    gid, nb = blk.gid, blk.nb
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
        raw = {"n_obs": n_obs, "pilots": p.tolist(), "rho": rho, "ref_b": blk.ref(Hr),
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
        out[str(n_obs)] = g04.summarize(raw, snrs, blk.boots, ci_digits=4)
        wc = out[str(n_obs)]["worst_case_loss_db"]
        msg = (f"  mismatch {n_obs:2d} pilots ({time.time() - t0:.0f} s): worst loss vs own matched SNR: "
               + "  ".join(f"{v} {wc[v]['vs_own_matched']['loss_db']:.2f}" for v in ["lmmse"] + list(nets)
                           if wc[v]["vs_own_matched"]))
        if "mlp_blind" in nets:
            msg += f"; blind MLP vs matched LMMSE at most {wc['mlp_blind']['vs_matched_lmmse']['loss_db']:.2f}"
        log(msg)
    return out


def check_rehearsal_mismatch(out, fz, log):
    """Run 56 odd blocks: the grid must equal experiment 04's record (LMMSE loadings, the LMMSE and linear
    grids, and every seed's NMSE grid of every variant)."""
    rec = read_json(os.path.join(fz["mismatch"]["validation_run"], "run.json"))["metrics"]["grid"]
    pairs = []
    for dens, summ in out.items():
        g, r = summ["grid"], rec[dens]
        if {k: float(v) for k, v in g["lmmse_rho"].items()} != {k: float(v) for k, v in r["lmmse_rho"].items()}:
            raise RuntimeError(f"{dens} pilots: frozen LMMSE loadings differ from experiment 04's choice")
        pairs += [(f"{dens} lmmse", g["lmmse"]["nmse_db"], r["lmmse"]["nmse_db"]),
                  (f"{dens} linear", g["linear"]["nmse_db"], r["linear"]["nmse_db"])]
        for v in fz["mismatch"]["variants"]:
            if set(g[v]["per_seed_nmse_db"]) != set(r[v]["per_seed_nmse_db"]):
                raise RuntimeError(f"{dens} {v}: seeds differ from experiment 04's record")
            for sd, grid in g[v]["per_seed_nmse_db"].items():
                pairs.append((f"{dens} {v} seed {sd}", grid, r[v]["per_seed_nmse_db"][sd]))
    return max_abs_diff(pairs, "mismatch", log)


# ----------------------------------------------------------------------------- joint setting
def load_spacefreq(path, run=None, scale=None, confirm=False, thr=-30.0, block_len=50):
    """Experiment 03's "spacefreq" vectors (gmm_feasibility.load, same arithmetic): snapshots whose 32 antennas
    all pass the screening, 16 of the 64 subcarriers (entry = antenna * 16 + grid subcarrier).

    run=None: the fit vectors (runs 0053-0055) and the global scale computed from them.
    run=56/57: only that run's vectors, multiplied by the given fit scale."""
    if run == TEST_RUN and not confirm:
        raise PermissionError(f"run {TEST_RUN:04d} is the held-out test run; it needs confirm=True")
    z = np.load(path)
    csi, q, runs, index = z["csi"], z["quality"], z["run"], z["index"]
    good = (q <= thr).all(1)
    sc = pilot_positions(64, 16)

    def vec(m):
        idx = np.flatnonzero(m & good)
        return csi[idx][:, :, sc].reshape(len(idx), -1).astype(np.complex128), idx
    if run is None:
        Xf, _ = vec(np.isin(runs, (53, 54, 55)))
        s = 1 / np.sqrt(np.mean(np.abs(Xf) ** 2))
        return dict(Xf=Xf * s, scale=float(s))
    Xe, se = vec(runs == run)
    n_run = int(np.sum(runs == run))
    return dict(Xe=Xe * scale, block=index[se] // block_len, n_run_snapshots=n_run,
                screened_fraction=round(1 - len(se) / n_run, 4))


def prepare_joint(fz, rec06, mlp, F, log):
    """Pre-flight of the joint part on the training runs only: consistency with the 06 / 07 records, the
    LMMSE objects and joint MLPs, and the six GMM refits checked against experiment 06 (labels and model)."""
    j = fz["joint"]
    Xf = F["Xf"]
    n, d = Xf.shape
    cfg06 = rec06["config"]
    if abs(rec06["metrics"]["data"]["scale"] - F["scale"]) > 1e-12 * F["scale"] or rec06["metrics"]["data"]["n_fit"] != n:
        raise ValueError("fit data differ from experiment 06's")
    if cfg06["em_max"] < n or cfg06["gmm_rho"] != 0.0:
        raise ValueError("experiment 06 used an EM subsample or a GMM loading; this script assumes neither")
    for p, (rec, _) in mlp.items():
        if abs(rec["config"]["scale"] - F["scale"]) > 1e-12 * F["scale"] or rec["config"]["n_fit"] != n:
            raise ValueError(f"{p}: fit data differ from experiment 07's")
    pats = j07.pattern_dict()
    Cg = j07.global_cov(Xf)                    # LMMSE and residual base: cov(all fit vectors) + 1e-6 loading (03)
    Cg_fit = g03.cov(Xf)                       # GMM: k-means phase reference and shrinkage target (06)
    P = {p: pats[p] for p in j["patterns"]}
    M = {p: np.setdiff1d(np.arange(d), P[p]) for p in j["patterns"]}
    rho, lms, nets = {}, {}, {}
    for p in j["patterns"]:
        rec07, paths = mlp[p]
        lms[p] = LMMSE(np.zeros(d, complex), Cg, P[p])
        for s in j["snrs"]:
            r = rec07["metrics"]["nmse"][p][str(s)]["lmmse"]["rho"]
            if r != rec06["metrics"]["lmmse"][p][str(s)]["rho"]:
                raise ValueError(f"{p} {s} dB: experiments 06 and 07 chose different LMMSE loadings")
            rho[p, s] = r
        for v in j["mlp_variants"]:
            for seed in j["mlp_seeds"]:
                pred, meta = j07.load_joint_predictor(paths[f"{p.replace(' x ', '-')}_{v}_seed{seed}.npz"], lms[p])
                if abs(meta["scale"] - F["scale"]) > 1e-12 * F["scale"]:
                    raise ValueError(f"{p} {v} seed {seed}: trained with scale {meta['scale']}, data {F['scale']}")
                nets[p, v, seed] = pred
    gmms, fits = {}, {}
    for seed in j["gmm_seeds"]:                # each seed's own validation choice per row
        rows06 = rec06["metrics"]["per_seed"][str(seed)]["rows"]
        for name in sorted({rows06[p][str(s)]["chosen"] for p in j["patterns"] for s in j["snrs"]}):
            K, alpha = next((r["K"], r["alpha"]) for pr in rows06.values() for r in pr.values() if r["chosen"] == name)
            want = rec06["metrics"]["per_seed"][str(seed)]["fits"][name]
            t = time.time()
            for attempt in range(1, KMEANS_TRIES + 1):
                labels = g06.kmeans_labels(Xf, Cg_fit, K, seed)
                if g06.digest(labels) == want["kmeans_labels_sha256"]:
                    break
                log(f"  GMM seed {seed} {name}: k-means labelling differs from experiment 06 (attempt {attempt}); retrying")
            else:
                raise RuntimeError(f"GMM seed {seed} {name}: k-means did not reproduce experiment 06's labelling in "
                                   f"{KMEANS_TRIES} attempts on the training data")
            pi, C, info = g06.fit_em(Xf, K, alpha, labels, Cg_fit, iters=cfg06["em_iters_cap"], tol=cfg06["em_tol"])
            if info["model_sha256"] != want["model_sha256"]:
                raise RuntimeError(f"GMM seed {seed} {name}: refit hash {info['model_sha256']} differs from experiment "
                                   f"06's {want['model_sha256']}; the frozen model is not reproduced")
            gmms[seed, name] = (pi, C)
            fits[f"seed{seed} {name}"] = {"model_sha256": info["model_sha256"], "kmeans_labels_sha256": g06.digest(labels),
                                          "kmeans_attempts": attempt, "em_iterations": info["iters"],
                                          "refit_s": round(time.time() - t, 1)}
            log(f"  GMM seed {seed} {name}: refit in {time.time() - t:.0f} s ({info['iters']} EM iterations, "
                f"k-means attempt {attempt}), labels and model equal experiment 06's")
    return dict(P=P, M=M, rho=rho, lms=lms, nets=nets, gmms=gmms, Cg=Cg, fits=fits,
                choice={(seed, p, s): rec06["metrics"]["per_seed"][str(seed)]["rows"][p][str(s)]["chosen"]
                        for seed in j["gmm_seeds"] for p in j["patterns"] for s in j["snrs"]})


def evaluate_joint(fz, prep, E, mask, log):
    """Joint LMMSE, GMM and joint MLPs on the vectors `mask` of E["Xe"]; estimates are reduced at once."""
    j = fz["joint"]
    X = E["Xe"]
    blk = Blocks(E["block"][mask])
    out = {}
    for p in j["patterns"]:
        P, M = prep["P"][p], prep["M"][p]
        Hm = X[mask][:, M]
        ref_b = blk.ref(Hm)
        out[p] = {}
        for s in j["snrs"]:
            nv = 10 ** (-s / 10)
            Y = add_noise(X[:, P], nv, seed=7919 * len(P) + s)[mask]   # same draw as experiments 03 / 06 / 07
            err = {"lmmse": blk.err(g03.mix_estimate(Y, P, np.ones(1), prep["Cg"][None], nv, prep["rho"][p, s])[:, M], Hm)}
            for seed in j["gmm_seeds"]:
                pi, C = prep["gmms"][seed, prep["choice"][seed, p, s]]
                err[f"gmm@seed{seed}"] = blk.err(g03.mix_estimate(Y, P, pi, C, nv)[:, M], Hm)
            for v in j["mlp_variants"]:
                for seed in j["mlp_seeds"]:
                    err[f"{v}@seed{seed}"] = blk.err(prep["nets"][p, v, seed](Y, s), Hm)
            fams = {"gmm": j["gmm_seeds"], **{v: j["mlp_seeds"] for v in j["mlp_variants"]}}
            for f, sds in fams.items():                                     # seed averages
                err[f"{f}@mean"] = np.mean([err[f"{f}@seed{sd}"] for sd in sds], axis=0)
            row, compare = summarize_row(blk, ref_b, err, {"lmmse_rho": prep["rho"][p, s]})
            for v in j["mlp_variants"]:                                     # positive = GMM better
                row[f"gmm@mean_gain_over_{v}@mean_db"] = compare(f"{v}@mean", "gmm@mean")
            out[p][str(s)] = row
            msg = (f"  {p:>12s} {s:2d} dB: LMMSE {row['lmmse']['nmse_db']:7.2f} | gain over LMMSE "
                   + " ".join(f"{f} {row[f'{f}@mean_gain_over_lmmse_db']['value']:+.2f}" for f in fams))
            for v in j["mlp_variants"]:
                g = row[f"gmm@mean_gain_over_{v}@mean_db"]
                msg += f" | GMM gain over {v} {g['value']:+.2f} [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}]"
            log(msg)
    return out, blk.nb


def check_rehearsal_joint(out, fz, rec06, mlp, log):
    """Run 56 odd blocks: every joint number must equal the experiment 06 / 07 records (to rounding)."""
    j = fz["joint"]
    pairs = []
    for p, rows in out.items():
        for s, row in rows.items():
            pairs.append((f"{p} {s} lmmse", row["lmmse"]["nmse_db"], rec06["metrics"]["lmmse"][p][s]["nmse_db"]))
            for seed in j["gmm_seeds"]:
                ref = rec06["metrics"]["per_seed"][str(seed)]["rows"][p][s]["nmse_db"]
                pairs.append((f"{p} {s} gmm seed {seed}", row[f"gmm@seed{seed}"]["nmse_db"], ref))
            r07 = mlp[p][0]["metrics"]["nmse"][p][s]
            for v in j["mlp_variants"]:
                for seed in j["mlp_seeds"]:
                    pairs.append((f"{p} {s} {v} seed {seed}", row[f"{v}@seed{seed}"]["nmse_db"],
                                  r07[v]["seeds"][str(seed)]["nmse_db"]))
    return max_abs_diff(pairs, "joint", log)


# ----------------------------------------------------------------------------- provenance and guards
CODE_SPEC = [":(glob)src/**/*.py", ":(glob)experiments/**/*.py", ":(glob)tools/**/*.py"]   # code that can change a number


def code_dirty():
    """Uncommitted or untracked Python code under src/, experiments/ or tools/ (None if git is unavailable)."""
    out = _git("status", "--porcelain", "--untracked-files=all", "--", *CODE_SPEC)
    return None if out is None else bool(out)


def record_paths(fz):
    return [f"{r}/run.json" for r in frozen_records(fz)]


def computational(fz):
    """The parts of frozen.json that decide a number (status, claims and notes do not)."""
    return {k: fz[k] for k in ("main", "mismatch", "joint")}


def tracked_clean(rel):
    return _git("ls-files", rel) == rel and not _git("status", "--porcelain", "--", rel)


def same_inputs(old_start, old_frozen, fz, data_sha):
    """None if a run that started at old_start used the code, frozen records, frozen choices and data of now;
    otherwise the reason why not."""
    if old_start.get("data", {}).get("sha256") != data_sha:
        return "data file differs"
    if computational(old_frozen) != computational(fz):
        return "frozen choices differ"
    if not git_ok("diff", "--quiet", old_start["git_commit"], _git("rev-parse", "HEAD"), "--",
                  *CODE_SPEC, *record_paths(fz)):
        return f"code or frozen records changed since {old_start['git_commit'][:7]}"
    return None


def earlier_attempts():
    return sorted(glob.glob(os.path.join(ROOT, "results", f"*{TEST_TAG}*"))
                  + glob.glob(os.path.join(ROOT, "data", "logs", f"*{TEST_TAG}*")))


def find_rehearsal(fz, env, data_sha):
    """A committed, passing three-part odd-block rehearsal that ran on clean code which did not change during
    the run, and whose code, frozen records, frozen choices, data file and environment equal the current ones."""
    reasons = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*_rehearsal-0056-odd", "run.json"))):
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        c, m, tag = r["config"], r["metrics"], rel.split("/")[1]
        st, en = c.get("start", {}), c.get("end", {})
        if list(c.get("parts", [])) != list(PARTS) or not c.get("odd_only"):
            why = "not a three-part odd-block rehearsal"
        elif not (st.get("code_dirty") is False and en.get("code_dirty") is False
                  and c.get("code_changed_during_run") is False and st.get("records_clean") is True):
            why = "code or frozen records were not clean, or changed during the run"
        elif not all(k in m and m[k] <= TOL_DB for k in ("rehearsal_main_max_abs_diff_db",
                                                       "rehearsal_mismatch_max_abs_diff_db",
                                                       "rehearsal_joint_max_abs_diff_db")):
            why = "a reproduction check is missing or failed"
        elif not tracked_clean(rel):
            why = "the rehearsal record is not committed (or is modified)"
        elif c.get("env") != env:
            why = f"environment {c.get('env')} differs from {env}"
        else:
            why = same_inputs(st, c["frozen"], fz, data_sha)
        if why is None:
            return rel.rsplit("/", 1)[0], reasons
        reasons.append(f"{tag}: {why}")
    return None, reasons


# ----------------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--data", default=DEFAULT_NPZ)
    ap.add_argument("--run", type=int, required=True, help="56 = rehearsal on validation, 57 = THE test")
    ap.add_argument("--odd-only", action="store_true", help="rehearsal: only the odd blocks of run 56 (reproducible)")
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS,
                    help="rehearsal only: evaluate a subset (the test always evaluates every part)")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--after-failed-attempt", metavar="REASON",
                    help="test only: an earlier attempt crashed (see results/ and data/logs/); say what happened")
    a = ap.parse_args()
    t0 = time.time()
    stamp = time.strftime("%Y%m%d-%H%M")
    parts = [p for p in PARTS if p in a.parts]
    test = a.run == TEST_RUN
    name = TEST_TAG if test else ("rehearsal-0056" + ("-odd" if a.odd_only else "-full")
                                  + ("" if parts == list(PARTS) else "-" + "-".join(parts)))
    logdir = os.path.join(ROOT, "data", "logs")
    state = {"logfile": None, "buffer": []}            # the log file is opened only once an attempt has started

    def log(s):
        print(s, flush=True)
        if state["logfile"] is None:
            state["buffer"].append(s)
        else:
            state["logfile"].write(s + "\n")
            state["logfile"].flush()

    def open_log():
        os.makedirs(logdir, exist_ok=True)
        state["logfile"] = open(os.path.join(logdir, f"{stamp}_{name}.log"), "a", encoding="utf-8")
        for s in state["buffer"]:
            state["logfile"].write(s + "\n")
        state["logfile"].flush()

    # ---- checks that read no data
    if test and (not a.confirm_test or a.odd_only or parts != list(PARTS)):
        sys.exit("the test run needs --confirm-test, no --odd-only and every part")
    if not test and a.run != 56:
        sys.exit("only run 56 (rehearsal) or 57 (test) are meaningful here")
    if not test and (a.confirm_test or a.after_failed_attempt):
        sys.exit("--confirm-test / --after-failed-attempt belong to the test run only")
    env = environment()
    if env["openblas_threads"] != [4]:
        sys.exit(f"OpenBLAS runs with {env['openblas_threads']} threads, not 4; the frozen GMMs need 4")
    fz, base, seeds, rec06, mlp = load_frozen()                         # SHA-256 of every weight file
    if rec06.get("numpy") != env["numpy"] or str(rec06.get("blas_threads")) != "4":   # GMM refits are bit-exact
        sys.exit(f"experiment 06 ran with numpy {rec06.get('numpy')} / {rec06.get('blas_threads')} BLAS threads; "
                 f"now numpy {env['numpy']} / 4: the GMM refits would not reproduce its models")
    if not os.path.exists(a.data):
        sys.exit(f"data file {a.data} not found")
    data_sha = sha256(a.data)
    config = {"run": a.run, "odd_only": a.odd_only, "parts": parts, "frozen": fz, "bootstrap_B": B_TEST,
              "merge_for_ci95_200snap": MERGE, "env": env,
              "start": {"git_commit": _git("rev-parse", "HEAD"), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "code_dirty": code_dirty(),
                        "records_clean": all(tracked_clean(r) for r in record_paths(fz)),
                        "frozen_json_committed_blob": _git("rev-parse", f"HEAD:{FROZEN_REL}"),
                        "frozen_json_clean": tracked_clean(FROZEN_REL),
                        "data": {"path": a.data, "bytes": os.path.getsize(a.data), "sha256": data_sha}}}
    if test:
        st = config["start"]
        if st["code_dirty"] is not False or not st["records_clean"] or not st["frozen_json_clean"]:
            sys.exit("refusing: commit the code, frozen.json and every record it names first")
        if fz.get("status") != "FROZEN":
            sys.exit("refusing: frozen.json status is not FROZEN")
        done = glob.glob(os.path.join(ROOT, "results", f"*_{TEST_TAG}", "run.json"))
        if done:
            sys.exit(f"refusing: the test run was already evaluated: {done}")
        rehearsal, why = find_rehearsal(fz, env, data_sha)
        if rehearsal is None:
            sys.exit("refusing: no usable rehearsal of this code, these frozen choices, this data and environment:\n  "
                     + "\n  ".join(why or ["no results/*_rehearsal-0056-odd record"]))
        earlier = earlier_attempts()
        if earlier and not a.after_failed_attempt:
            sys.exit(f"refusing: earlier test attempts exist: {earlier}\nif an attempt crashed, rerun with "
                     f"--after-failed-attempt \"<what happened>\"; that is written into the record")
        if a.after_failed_attempt and not earlier:
            sys.exit("--after-failed-attempt given, but no earlier attempt exists")
        for sj in glob.glob(os.path.join(ROOT, "results", f"*_{TEST_TAG}-started", "started.json")):
            with open(sj, encoding="utf-8") as f:
                old = json.load(f)
            why_old = same_inputs(old["start"], old["frozen"], fz, data_sha)
            if why_old:
                sys.exit(f"refusing: {sj}: {why_old}; nothing may change after an attempt that may have seen "
                         f"test numbers")
        config.update(rehearsal=rehearsal, after_failed_attempt=a.after_failed_attempt,
                      previous_attempts=[os.path.relpath(x, ROOT).replace(os.sep, "/") for x in earlier])
    else:
        open_log()
    log(f"{name}: {' '.join(sys.argv)}; env {env}")

    marker, metrics = None, {"definitions": DEFINITIONS}
    try:
        # ---- pre-flight on the training runs only (nothing of the evaluated run is extracted yet)
        F = prep = None
        if "joint" in parts:
            t = time.time()
            F = load_spacefreq(a.data)
            log(f"[pre-flight] joint fit vectors {F['Xf'].shape}; checking records, MLPs and the GMM refits")
            prep = prepare_joint(fz, rec06, mlp, F, log)
            config["joint_preflight"] = {"fits": prep["fits"], "runtime_s": round(time.time() - t, 1)}
        if test:                                        # every check passed: the attempt starts now
            marker = os.path.join(ROOT, "results", f"{stamp}_{a.member}_{TEST_TAG}-started")
            os.makedirs(marker, exist_ok=False)
            with open(os.path.join(marker, "started.json"), "w", encoding="utf-8") as f:
                json.dump({"note": "an attempt at THE test on run 0057 started here; commit this folder",
                           "command": " ".join(sys.argv), **config}, f, indent=1)
            open_log()
            log(f"[test] attempt marker {marker}")

        partial = os.path.join(logdir, f"{stamp}_{name}_partial.json")

        def keep():   # a crash in a later part must not lose the numbers already computed (data/ is not in git)
            with open(partial + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"config": config, "metrics": metrics}, f, indent=1)
            os.replace(partial + ".tmp", partial)

        check = a.odd_only and not test
        if not test and not a.odd_only:
            log("[full-size dry run] all blocks of run 56: the validation records cover only the odd blocks, "
                "so no reproduction check is possible here")
        if "joint" in parts:
            t = time.time()
            E = load_spacefreq(a.data, a.run, F["scale"], confirm=test)
            mask = (E["block"] % 2 == 1) if a.odd_only else np.ones(len(E["Xe"]), bool)
            log(f"[joint] run {a.run:04d}: evaluating {int(mask.sum())} snapshots; {E['screened_fraction']:.2%} of "
                f"the run's {E['n_run_snapshots']} snapshots screened out (not all 32 antennas ok)")
            metrics["joint"], nb = evaluate_joint(fz, prep, E, mask, log)
            config["joint_data"] = {"n_fit": int(len(F["Xf"])), "n_vectors": int(mask.sum()), "n_blocks": nb,
                                    "dim": int(E["Xe"].shape[1]), "n_run_snapshots": E["n_run_snapshots"],
                                    "screened_fraction": E["screened_fraction"], "runtime_s": round(time.time() - t, 1)}
            keep()
            del E, F, prep, mask
            gc.collect()
            if check:
                metrics["rehearsal_joint_max_abs_diff_db"] = check_rehearsal_joint(metrics["joint"], fz, rec06, mlp, log)
        if "main" in parts or "mismatch" in parts:
            d = load_test(a.data, test_run=a.run, confirm=test)
            mask = d.odd if a.odd_only else np.ones(len(d.Ht), bool)
            log(f"[main] run {a.run:04d}: fit {len(d.Hf)}, evaluating {int(mask.sum())} vectors; "
                f"screened fraction {d.screened_fraction}")
            config["main_data"] = {"n_fit": int(len(d.Hf)), "n_vectors": int(mask.sum()),
                                   "screened_fraction": d.screened_fraction}
            if len(d.Hf) != base["metrics"]["summary"]["fit_vectors"]:
                raise RuntimeError("main-line fit set differs from the baselines record")
            if "main" in parts:
                t = time.time()
                metrics["main"], config["main_data"]["n_blocks"] = evaluate_main(fz, base, seeds, d, mask, log)
                config["main_data"]["runtime_s"] = round(time.time() - t, 1)
                keep()
                if check:
                    metrics["rehearsal_main_max_abs_diff_db"] = check_rehearsal_main(metrics["main"], fz, base, seeds, log)
            if "mismatch" in parts:
                t = time.time()
                log("[mismatch] true x assumed SNR grid with the frozen designs")
                metrics["mismatch"] = evaluate_mismatch(fz, base, seeds, d, mask, log)
                config["mismatch_runtime_s"] = round(time.time() - t, 1)
                keep()
                if check:
                    metrics["rehearsal_mismatch_max_abs_diff_db"] = check_rehearsal_mismatch(metrics["mismatch"], fz, log)
            del d, mask
            gc.collect()

        config["runtime_s"] = round(time.time() - t0, 1)
        end_commit = _git("rev-parse", "HEAD")
        config["end"] = {"git_commit": end_commit, "code_dirty": code_dirty(),
                         "records_clean": all(tracked_clean(r) for r in record_paths(fz))}
        config["code_changed_during_run"] = bool(
            config["end"]["code_dirty"] is not False or not config["end"]["records_clean"]
            or not git_ok("diff", "--quiet", config["start"]["git_commit"], end_commit, "--",
                          *CODE_SPEC, *record_paths(fz)))
        if config["code_changed_during_run"]:
            log("WARNING: code or frozen records changed while this ran; config.start holds the state it ran with")
        save_run(name, a.member, config, metrics,
                 note=("THE single evaluation on test run 0057 with frozen choices" if test
                       else "rehearsal of the test code path on validation run 0056"))
        os.remove(partial)                                              # the full record is saved
        log(f"done in {time.time() - t0:.0f} s")
    except BaseException as exc:
        if state["logfile"] is not None:
            log("FAILED:\n" + traceback.format_exc())
        if marker is not None:
            with open(os.path.join(marker, "crashed.json"), "w", encoding="utf-8") as f:
                json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "exception": repr(exc),
                           "traceback": traceback.format_exc(),
                           "parts_finished": [k for k in PARTS if k in metrics]}, f, indent=1)
        raise
    finally:
        if state["logfile"] is not None:
            state["logfile"].close()


if __name__ == "__main__":
    main()
