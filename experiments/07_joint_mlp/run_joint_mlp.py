"""Joint antenna x subcarrier neural estimators versus joint LMMSE (validation run 0056).

Question: can a generic learned estimator exploit the joint antenna x frequency structure that the
GMM of experiment 03 exploits? Setting "spacefreq" of experiments/03 (its loader is imported, not
copied): 32 antennas x 16 subcarriers (every ~4th of the 64-block) = 512 entries per snapshot,
snapshots whose 32 antennas all pass the screening, one global scale from the training data.
Pilot patterns: "32ant x 4sc" (128 pilots, 384 withheld) and "8ant x 4sc" (32 pilots, 480 withheld).

Methods (input: noisy pilots y and the pilot SNR; output: the withheld entries)
  lmmse      joint LMMSE, zero mean, covariance of ALL training snapshots (+1e-6 loading as in 03),
             loading rho chosen on the even blocks for every SNR (= "lmmse" in results_spacefreq.json)
  jmlp       MLP [Re y, Im y, snr/30] -> [Re, Im] of the withheld entries
  jresidual  the same MLP learns a correction to a per-sample joint LMMSE base whose loading
             rho_base is ONE value per pattern (chosen on the even blocks, mean NMSE over the SNRs)
Training uses the ~15.4k training snapshots only: pilot SNR ~ U[0,30] dB with fresh noise every
batch, a random common phase per snapshot (default; the calibrated CSI has a random common phase,
so this is a symmetry of the data, see README), Adam, batch 256, learning-rate steps as in
experiments/02, early stopping on the even validation blocks (fixed noise, seeds 7000+snr). The best
weights of every model are saved as artifacts (data/runs/<run folder>/, SHA-256 in run.json).

Reported on the odd blocks with the noise of experiments/03 (seed 7919*len(P)+snr), so the LMMSE
numbers and the GMM reference of results_spacefreq.json are on identical inputs. Gains are dB better
than the per-SNR LMMSE, paired block-bootstrap 95% CI (B=1000), per seed and for the seed ensemble
(per-block errors averaged over the seeds). The test run 0057 is never loaded.

    python experiments/07_joint_mlp/run_joint_mlp.py --member <name> --smoke      # code test, ~1.5 min
    python experiments/07_joint_mlp/run_joint_mlp.py --member <name> --check-lmmse   # LMMSE = exp. 03? ~2 min
    python experiments/07_joint_mlp/run_joint_mlp.py --member <name>              # full run
    python experiments/07_joint_mlp/run_joint_mlp.py --member test --synthetic --smoke   # no data file
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # default thread count is far slower on small matrices
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP03 = os.path.join(ROOT, "experiments", "03_gmm_feasibility")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, EXP03)
import gmm_feasibility as g03  # noqa: E402  (load, cov, mix_estimate, RHOS, SNRS of experiment 03)
from dataset import DEFAULT_NPZ, bootstrap_ci  # noqa: E402
from estimators import LMMSE, add_noise, nmse_db, pilot_positions  # noqa: E402
from mlp_numpy import MLP  # noqa: E402
from neural import features  # noqa: E402
from runlog import save_run  # noqa: E402

SNRS = list(g03.SNRS)            # [0, 10, 20, 30], as experiment 03
RHOS = list(g03.RHOS)            # LMMSE loading grid of experiment 03 (x mean pilot power)
PATTERNS = ["32ant x 4sc", "8ant x 4sc"]
VARIANTS = {"jmlp": False, "jresidual": True}          # name: learns a correction to the LMMSE base
LEGEND = {"jmlp": "Joint MLP", "jresidual": "Joint LMMSE + residual MLP"}
STYLE_KEY = {"jmlp": "mlp", "jresidual": "residual"}   # colour / marker family in src/style.py
REF03 = os.path.join(EXP03, "results_spacefreq.json")
SMOKE_FIT, SMOKE_VAL, SMOKE_EPOCHS = 3000, 1000, 3


# ----------------------------------------------------------------------------- data
def pattern_dict():
    """The two pilot patterns of experiment 03 (entry index = antenna * 16 + grid subcarrier)."""
    sc_p = pilot_positions(16, 4)
    grid = lambda ants: (ants[:, None] * 16 + sc_p[None, :]).ravel()
    return {"32ant x 4sc": grid(np.arange(32)), "8ant x 4sc": grid(np.arange(0, 32, 4))}


def smoke_data(D, n_fit=SMOKE_FIT, n_val=SMOKE_VAL):
    """Code test on the TRAINING runs only: the first 70% of the training snapshots fit, the next
    ones stand in for validation (pseudo-blocks of 50), so a smoke test never uses run 0056 (the
    experiment-03 loader still reads it into memory; it is dropped here)."""
    Xf = D["Xf"]
    cut = int(0.7 * len(Xf))
    fit, val = Xf[:min(cut, n_fit)].copy(), Xf[cut:cut + n_val].copy()
    block = np.arange(len(val)) // 50
    return dict(Xf=fit, Xv=val, sel=block % 2 == 0, rep=block % 2 == 1, block=block, pats=D["pats"],
                scale=D["scale"], source="smoke: training runs only (first 70% fit, next snapshots validate)")


def synthetic_data(n_fit=4000, n_val=1500, n_pos=8, n_paths=4, seed=0):
    """Test data with the real shapes (32 antennas x 16 subcarriers, no data file needed): a few paths
    per snapshot whose directions and delays depend on a hidden position cluster, so the channel is a
    mixture of low-rank Gaussians as in the measured data; random common phase per snapshot."""
    rng = np.random.default_rng(seed)
    ant, sc = np.arange(32), pilot_positions(64, 16)
    theta_c = rng.uniform(-np.pi / 3, np.pi / 3, n_pos)
    tau_c = rng.uniform(0, 6, (n_pos, n_paths))              # delays in samples of a 1024-point FFT
    prof = np.exp(-np.arange(n_paths) / 1.5)

    def draw(n):
        c = rng.integers(0, n_pos, n)
        th = theta_c[c][:, None] + 0.05 * rng.standard_normal((n, n_paths))
        tau = tau_c[c] + 0.3 * rng.standard_normal((n, n_paths))
        g = (rng.standard_normal((n, n_paths)) + 1j * rng.standard_normal((n, n_paths))) * np.sqrt(prof / 2)
        A = np.exp(1j * np.pi * ant[None, None, :] * np.sin(th)[:, :, None])      # (n, L, 32)
        F = np.exp(-2j * np.pi * sc[None, None, :] * tau[:, :, None] / 1024)       # (n, L, 16)
        H = np.einsum("nl,nla,nlf->naf", g, A, F).reshape(n, 512)
        return H * np.exp(2j * np.pi * rng.random(n))[:, None]
    Xf, Xv = draw(n_fit), draw(n_val)
    scale = 1 / np.sqrt(np.mean(np.abs(Xf) ** 2))
    block = np.arange(n_val) // 50
    return dict(Xf=Xf * scale, Xv=Xv * scale, sel=block % 2 == 0, rep=block % 2 == 1, block=block,
                pats=pattern_dict(), scale=float(scale), source="synthetic channels of the real shapes")


def global_cov(Xf):
    """Zero-mean covariance of all training snapshots plus 1e-6 loading, exactly as experiment 03."""
    d = Xf.shape[1]
    Cg = g03.cov(Xf)
    return Cg + 1e-6 * np.real(np.trace(Cg)) / d * np.eye(d)


# ----------------------------------------------------------------------------- models
class InPlaceAdamMLP(MLP):
    """src/mlp_numpy.MLP with the same backward pass and Adam update, done in place.

    For ~0.8M parameters the allocating update of MLP.step takes about half of a training step; this
    one gives the same parameters up to rounding (checked in the README tests). Weights, save() and
    MLP.from_file() are unchanged, so saved models load with the plain MLP class.
    """

    def step(self, cache, dout, lr, b1=0.9, b2=0.999, eps=1e-8):
        acts, pre = cache
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (pre[i - 1] > 0)
        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        if not hasattr(self, "_tmp"):
            self._tmp = [np.empty_like(p) for p in self.params]
        for p, g, m, v, tmp in zip(self.params, gW + gb, self.m, self.v, self._tmp):
            np.multiply(g, 1 - b1, out=tmp)
            m *= b1
            m += tmp                                   # m = b1 m + (1 - b1) g
            np.multiply(g, g, out=tmp)
            tmp *= 1 - b2
            v *= b2
            v += tmp                                   # v = b2 v + (1 - b2) g^2
            np.divide(v, c2, out=tmp)
            np.sqrt(tmp, out=tmp)
            tmp += eps
            np.divide(m, tmp, out=tmp)
            tmp *= lr / c1
            p -= tmp                                   # p -= lr (m / c1) / (sqrt(v / c2) + eps)


def check_inplace_adam():
    """InPlaceAdamMLP copies the backward pass and Adam of MLP.step; refuse to train if a later edit of
    src/mlp_numpy.py makes the two differ (a few steps on a tiny network, ~10 ms)."""
    rng = np.random.default_rng(0)
    ref, new = MLP([9, 16, 16, 6], seed=1), InPlaceAdamMLP([9, 16, 16, 6], seed=1)
    for _ in range(5):
        X, T = rng.standard_normal((8, 9)), rng.standard_normal((8, 6))
        for net in (ref, new):
            out, cache = net.forward(X)
            net.step(cache, 2 * (out - T) / out.size, 1e-3)
    err = max(float(np.max(np.abs(p - q))) for p, q in zip(ref.params, new.params))
    if err > 1e-12:
        raise RuntimeError(f"InPlaceAdamMLP differs from mlp_numpy.MLP.step by {err:.1e}; update the subclass")


def make_joint_predictor(variant, net, lm, M, rho_base):
    """Closure (Y, snr_db[, base]) -> estimate of the withheld entries M (complex, (n, |M|)).

    snr_db is the pilot SNR the receiver assumes; jresidual also uses it in its LMMSE base.
    `base` (optional) is the precomputed LMMSE base on M for the same Y and SNR.
    """
    residual, m = VARIANTS[variant], len(M)

    def predict(Y, snr_db, base=None):
        s = np.broadcast_to(np.asarray(snr_db, float), (len(Y),))
        out = net.predict(features(Y, s))
        r = out[:, :m] + 1j * out[:, m:]
        if residual:
            r = r + (lm.apply_per_sample(Y, 10 ** (-s / 10), rho_base)[:, M] if base is None else base)
        return r
    return predict


def load_joint_predictor(path, lm):
    """Rebuild a saved model; lm = LMMSE(zeros(512), Cg, P) on the same training statistics."""
    net, meta = MLP.from_file(path)
    if list(meta["pilots"]) != [int(x) for x in lm.p]:
        raise ValueError(f"{path}: pilots do not match the LMMSE pilots")
    M = np.asarray(meta["withheld"], int)
    if not np.array_equal(M, np.setdiff1d(np.arange(lm.C.shape[0]), lm.p)):
        raise ValueError(f"{path}: withheld entries do not match")
    if VARIANTS[meta["variant"]] and "cg_fro" in meta:   # the residual base depends on Cg itself
        fro = float(np.linalg.norm(lm.C))
        if abs(fro - meta["cg_fro"]) > 1e-9 * fro:
            raise ValueError(f"{path}: LMMSE covariance differs from the training one "
                             f"(Frobenius norm {fro:.10g} vs {meta['cg_fro']:.10g}); rebuild Cg with global_cov")
    return make_joint_predictor(meta["variant"], net, lm, M, float(meta["rho_base"])), meta


def choose_rho_base(lm, Xv, sel, P, M):
    """ONE loading for the residual base: mean NMSE over SNRS on the even blocks (noise of 03)."""
    Hs = Xv[sel][:, M]
    Ys = {s: add_noise(Xv[:, P], 10 ** (-s / 10), seed=7919 * len(P) + s)[sel] for s in SNRS}
    score = {r: float(np.mean([nmse_db(lm.apply_per_sample(Ys[s], 10 ** (-s / 10), r)[:, M], Hs) for s in SNRS]))
             for r in RHOS}
    return min(RHOS, key=score.get), {str(r): round(v, 4) for r, v in score.items()}


def train(variant, Xf, Xs, P, M, lm, rho_base, a, seed, log):
    """Train one model; early stopping on the even blocks Xs (mean NMSE over SNRS, noise seeds 7000+snr)."""
    residual = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    net = InPlaceAdamMLP([2 * len(P) + 1] + list(a.hidden) + [2 * len(M)], seed=seed)
    if residual:   # zero output layer: training starts exactly at the LMMSE base, not at base + random output
        net.W[-1][...] = 0.0
    predict = make_joint_predictor(variant, net, lm, M, rho_base)
    Ysel = {s: add_noise(Xs[:, P], 10 ** (-s / 10), seed=7000 + s) for s in SNRS}
    Bsel = {s: lm.apply_per_sample(Ysel[s], 10 ** (-s / 10), rho_base)[:, M] if residual else None for s in SNRS}
    Ts = Xs[:, M]
    steps = len(Xf) // a.batch
    val_db = lambda: float(np.mean([nmse_db(predict(Ysel[s], s, Bsel[s]), Ts) for s in SNRS]))
    t0 = time.time()
    best = val_db()                     # epoch 0 = untrained start (for jresidual: the LMMSE base itself)
    best_state, best_ep, hist = net.state(), 0, [round(best, 3)]
    for ep in range(a.epochs):
        lr = 1e-3 if ep < 0.6 * a.epochs else (3e-4 if ep < 0.85 * a.epochs else 1e-4)   # as experiments/02
        perm = rng.permutation(len(Xf))
        for b in range(steps):
            h = Xf[perm[b * a.batch:(b + 1) * a.batch]]
            if a.phase_aug:
                h = h * np.exp(2j * np.pi * rng.random(len(h)))[:, None]
            s = rng.uniform(0, 30, len(h))
            nv = 10 ** (-s / 10)
            Y = h[:, P] + np.sqrt(nv / 2)[:, None] * (rng.standard_normal((len(h), len(P)))
                                                     + 1j * rng.standard_normal((len(h), len(P))))
            T = h[:, M]
            if residual:
                T = T - lm.apply_per_sample(Y, nv, rho_base)[:, M]
            out, cache = net.forward(features(Y, s))
            diff = out - np.concatenate([T.real, T.imag], axis=1)
            net.step(cache, 2 * diff / diff.size, lr)
            if a.weight_decay:                                  # decoupled, weights only
                for W in net.W:
                    W -= lr * a.weight_decay * W
        val = val_db()
        hist.append(round(val, 3))
        if val < best:
            best, best_state, best_ep = val, net.state(), ep + 1
        if ep == 0 or (ep + 1) % a.log_every == 0 or ep + 1 == a.epochs:
            log(f"    {variant} seed {seed} epoch {ep + 1}/{a.epochs}: val {val:.3f} dB, best {best:.3f} "
                f"(epoch {best_ep})  ({time.time() - t0:.0f} s)")
    net.load(best_state)
    return net, predict, {"best_val_db": round(best, 3), "best_epoch": best_ep, "params": net.n_params(),
                          "train_s": round(time.time() - t0, 1), "val_db_by_epoch_from_0": hist}


def save_and_check(net, predict, path, lm, Ycheck, meta):
    """Save the weights, reload them and confirm the reloaded model predicts exactly the same."""
    net.save(path, **meta)
    again, _ = load_joint_predictor(path, lm)
    s = np.linspace(0, 30, len(Ycheck))
    if not np.array_equal(again(Ycheck, s), predict(Ycheck, s)):
        raise RuntimeError(f"reloaded model {path} does not reproduce the trained one")


# ----------------------------------------------------------------------------- evaluation
def ci95(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def evaluate(D, P, M, Cg, lm, rho_base, models, boots, rgid, n_blocks, log, pname):
    """Odd-block NMSE of LMMSE and every model; models = {variant: {seed: predictor}}."""
    Xv, sel, rep = D["Xv"], D["sel"], D["rep"]
    Hs, Hr = Xv[sel], Xv[rep]
    ref_b = np.bincount(rgid, np.sum(np.abs(Hr[:, M]) ** 2, 1), n_blocks)
    per_block = lambda E: np.bincount(rgid, np.sum(np.abs(E - Hr[:, M]) ** 2, 1), n_blocks)
    boot_db = lambda err_b: 10 * np.log10(err_b[boots].sum(1) / ref_b[boots].sum(1))
    out = {}
    for snr in SNRS:
        nv = 10 ** (-snr / 10)
        Y = add_noise(Xv[:, P], nv, seed=7919 * len(P) + snr)              # identical to experiment 03
        Ys, Yr = Y[sel], Y[rep]
        rho = min(RHOS, key=lambda r: nmse_db(g03.mix_estimate(Ys, P, np.ones(1), Cg[None], nv, r)[:, M], Hs[:, M]))
        E_l = g03.mix_estimate(Yr, P, np.ones(1), Cg[None], nv, rho)[:, M]
        bs_l, ci_l = bootstrap_ci(per_block(E_l), ref_b, boots)
        l_db = nmse_db(E_l, Hr[:, M])                          # unrounded; gains use unrounded values
        base = lm.apply_per_sample(Yr, nv, rho_base)[:, M]
        row = {"lmmse": {"nmse_db": round(l_db, 3), "rho": rho, "ci95": ci_l},
               "lmmse_rho_base": {"nmse_db": round(nmse_db(base, Hr[:, M]), 3), "rho": rho_base}}
        for v, by_seed in models.items():
            seeds, errs = {}, []
            for seed, predict in by_seed.items():
                E = predict(Yr, snr, base if VARIANTS[v] else None)
                err_b = per_block(E)
                errs.append(err_b)
                g = bs_l - boot_db(err_b)                      # paired: same resampled blocks
                e_db = nmse_db(E, Hr[:, M])
                seeds[str(seed)] = {"nmse_db": round(e_db, 3), "gain_over_lmmse_db": {
                    "value": round(l_db - e_db, 3), "ci95": ci95(g)}}
            err_e = np.mean(errs, axis=0)                      # seed ensemble: per-block errors averaged
            e_db = float(10 * np.log10(err_e.sum() / ref_b.sum()))
            gains = [x["gain_over_lmmse_db"]["value"] for x in seeds.values()]
            row[v] = {"seeds": seeds, "ensemble": {
                "nmse_db": round(e_db, 3), "ci95": bootstrap_ci(err_e, ref_b, boots)[1],
                "gain_over_lmmse_db": {"value": round(l_db - e_db, 3),
                                       "ci95": ci95(bs_l - boot_db(err_e))},
                "seed_gain_min_max_db": [min(gains), max(gains)]}}
        out[str(snr)] = row
        log(f"  {pname:>12s} {snr:2d} dB: LMMSE {row['lmmse']['nmse_db']:7.2f} (rho {rho:g}) | "
            + " | ".join(f"{v} {row[v]['ensemble']['gain_over_lmmse_db']['value']:+.2f} "
                         f"[{row[v]['ensemble']['gain_over_lmmse_db']['ci95'][0]:+.2f},"
                         f"{row[v]['ensemble']['gain_over_lmmse_db']['ci95'][1]:+.2f}]" for v in models))
    return out


def reference_exp03(patterns):
    """LMMSE and GMM numbers of results_spacefreq.json (experiment 03, one seed), not recomputed."""
    with open(REF03, encoding="utf-8") as f:
        r = json.load(f)["results"]
    return {p: {s: {"lmmse_nmse_db": r[p][s]["lmmse"]["nmse_db"], "lmmse_rho": r[p][s]["lmmse"]["rho"],
                    "gmm_chosen": r[p][s]["gmm"]["chosen"], "gmm_nmse_db": r[p][s]["gmm"]["nmse_db"],
                    "gmm_gain_db": r[p][s]["gmm"]["gain_over_lmmse_db"], "gmm_gain_ci95": r[p][s]["gmm"]["gain_ci95"]}
                for s in map(str, SNRS)} for p in patterns}


def check_against_exp03(nmse, ref):
    """Our LMMSE must reproduce experiment 03 on the real data (same data, noise, selection)."""
    diffs, rho_ok = [], True
    for p in nmse:
        for s in nmse[p]:
            diffs.append(abs(nmse[p][s]["lmmse"]["nmse_db"] - ref[p][s]["lmmse_nmse_db"]))
            rho_ok &= nmse[p][s]["lmmse"]["rho"] == ref[p][s]["lmmse_rho"]
    mx = round(float(max(diffs)), 4)
    return {"max_abs_diff_db": mx, "rho_all_equal": bool(rho_ok), "matches": bool(mx <= 0.0015 and rho_ok)}


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--data", default=DEFAULT_NPZ)
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--hidden", nargs="*", type=int, default=[512, 512])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--no-phase-aug", dest="phase_aug", action="store_false",
                    help="do not rotate each training snapshot by a random common phase")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="decoupled weight decay (0 = off)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help=f"code test: {SMOKE_FIT} training snapshots fit, {SMOKE_VAL} later TRAINING snapshots "
                         f"stand in for validation, {SMOKE_EPOCHS} epochs; nothing saved to results/")
    ap.add_argument("--synthetic", action="store_true",
                    help="test hook: synthetic channels of the real shapes instead of the data file; nothing saved")
    ap.add_argument("--check-lmmse", action="store_true",
                    help="pre-flight on the real data: only this script's joint LMMSE vs results_spacefreq.json "
                         "(same data and noise as the full run, no training); nothing saved")
    a = ap.parse_args(argv)
    if a.epochs < 1 or a.batch < 1 or a.boot < 1:
        ap.error("--epochs, --batch and --boot must be >= 1")
    if a.check_lmmse and a.smoke:
        ap.error("--check-lmmse compares with experiment 03 on validation run 0056; do not combine it with --smoke")
    log = lambda m: print(m, flush=True)
    t_start = time.time()
    check_inplace_adam()
    if a.smoke:
        a.epochs = min(a.epochs, SMOKE_EPOCHS)

    if a.synthetic:
        D = synthetic_data(*((SMOKE_FIT, SMOKE_VAL) if a.smoke else ()))
    else:
        D = g03.load(a.data, "spacefreq")
        D = smoke_data(D) if a.smoke else dict(D, source="runs 0053-0055 fit, validation run 0056")
    Xf, Xv, sel, rep = D["Xf"], D["Xv"], D["sel"], D["rep"]
    d = Xf.shape[1]
    rb, rgid = np.unique(D["block"][rep], return_inverse=True)
    boots = np.random.default_rng(0).integers(0, len(rb), (a.boot, len(rb)))
    log(f"[spacefreq] {D['source']}: fit {Xf.shape}, val {Xv.shape}, select {int(sel.sum())}, "
        f"report {int(rep.sum())} ({len(rb)} blocks); BLAS threads {os.environ.get('OPENBLAS_NUM_THREADS')}")
    Cg = global_cov(Xf)
    cg_fro = float(np.linalg.norm(Cg))          # fingerprint, checked when a residual model is reloaded
    ref = reference_exp03(a.patterns)

    nmse, rho_base_info, train_info = {}, {}, {}
    variants = [] if a.check_lmmse else a.variants
    weight_dir, artifacts = (tempfile.mkdtemp(prefix="joint_mlp_weights_") if variants else None), []
    for pname in a.patterns:
        P = np.asarray(D["pats"][pname])
        M = np.setdiff1d(np.arange(d), P)
        lm = LMMSE(np.zeros(d, complex), Cg, P)
        rho_base, scores = choose_rho_base(lm, Xv, sel, P, M)
        rho_base_info[pname] = {"chosen": rho_base, "mean_nmse_db_even_blocks": scores}
        log(f"  {pname}: {len(P)} pilots, {len(M)} withheld; residual base rho {rho_base:g} (even-block scores {scores})")
        models = {}
        for v in variants:
            models[v] = {}
            for seed in a.seeds:
                log(f"  training {v} seed {seed} for {pname}")
                net, models[v][seed], info = train(v, Xf, Xv[sel], P, M, lm, rho_base, a, seed, log)
                key = f"{pname.replace(' x ', '-')}_{v}_seed{seed}"
                train_info[key] = info
                path = os.path.join(weight_dir, key + ".npz")
                save_and_check(net, models[v][seed], path, lm, Xv[sel][:256, P],
                               dict(variant=v, residual=VARIANTS[v], setting="spacefreq", pattern=pname,
                                    pilots=P.tolist(), withheld=M.tolist(), dim=d, hidden=list(a.hidden),
                                    seed=seed, scale=D["scale"], rho_base=rho_base, best_epoch=info["best_epoch"],
                                    best_val_db=info["best_val_db"], epochs=a.epochs, batch=a.batch,
                                    phase_aug=a.phase_aug, weight_decay=a.weight_decay,
                                    features="[Re y, Im y, snr/30]", subcarriers=pilot_positions(64, 16).tolist(),
                                    output_init="zero (starts at the LMMSE base)" if VARIANTS[v] else "He",
                                    base="zero-mean joint LMMSE, Cg = cov(all training) + 1e-6 loading, "
                                         "LMMSE.apply_per_sample(Y, noise_var, rho_base)",
                                    cg_fro=cg_fro, data=D["source"]))
                artifacts.append(path)
        nmse[pname] = evaluate(D, P, M, Cg, lm, rho_base, models, boots, rgid, len(rb), log, pname)

    real_val = not (a.smoke or a.synthetic)     # real data, validation run 0056
    full = real_val and not a.check_lmmse
    check = check_against_exp03(nmse, ref) if real_val else None
    if check is not None:
        log(f"LMMSE vs experiment 03: {check}" + ("" if check["matches"] else "  <-- MISMATCH, investigate"))
    if a.check_lmmse:
        log(f"LMMSE pre-flight finished in {time.time() - t_start:.0f} s: "
            f"{'OK' if check and check['matches'] else 'NOT CONFIRMED'} (nothing saved)")
        return check
    fig_path = os.path.join(tempfile.mkdtemp(prefix="joint_mlp_"), "joint_gain_over_lmmse_val.png")
    files = []
    try:  # a plotting problem must never lose the metrics of a long run
        plot(nmse, ref, a, fig_path, None if full else D["source"])
        files = [fig_path]
    except Exception as e:
        log(f"plot failed ({e!r}); metrics are still saved")
    try:
        data_ref = os.path.relpath(a.data, ROOT).replace(os.sep, "/")
    except ValueError:  # different drive
        data_ref = os.path.basename(a.data)
    config = {"setting": "spacefreq (experiments/03 loader)", "data": "synthetic" if a.synthetic else data_ref,
              "data_source": D["source"], "threshold_db": -30.0, "block_len": 50, "scale": D["scale"], "dim": d,
              "n_fit": int(len(Xf)), "n_select": int(sel.sum()), "n_report": int(rep.sum()), "report_blocks": int(len(rb)),
              "patterns": {p: {"pilots": len(D["pats"][p]), "withheld": d - len(D["pats"][p])} for p in a.patterns},
              "variants": a.variants, "hidden": list(a.hidden), "epochs": a.epochs, "batch": a.batch, "seeds": a.seeds,
              "lr_schedule": "1e-3, 3e-4 from 60% of epochs, 1e-4 from 85%", "snr_train": "U[0,30] dB, fresh noise per batch",
              "phase_aug": a.phase_aug, "weight_decay": a.weight_decay, "jresidual_output_init": "zero",
              "optimizer": "Adam (in-place update, same as mlp_numpy.MLP.step)", "snrs_db": SNRS, "rho_grid": RHOS,
              "early_stopping": "even blocks, mean NMSE over snrs_db, noise seeds 7000+snr",
              "seed_ensemble": "per-block squared errors averaged over the seeds (average model), "
                               "not averaged predictions",
              "report_noise": "add_noise(Xv[:, P], nv, seed=7919*len(P)+snr) (experiment 03)",
              "bootstrap": {"B": a.boot, "seed": 0, "unit": "odd 50-snapshot blocks, paired"},
              "smoke": a.smoke, "synthetic": a.synthetic, "runtime_s": round(time.time() - t_start, 1)}
    metrics = {"nmse": nmse, "rho_base": rho_base_info, "training": train_info,
               "lmmse_check_vs_exp03": check, "reference_exp03": ref}
    if not full:
        log(f"{'synthetic ' if a.synthetic else ''}{'smoke ' if a.smoke else ''}test finished in "
            f"{time.time() - t_start:.0f} s (nothing saved to results/); figure {fig_path}; weights {weight_dir}")
        return config, metrics, files, artifacts
    save_run("joint-mlp-val", a.member, config, metrics, files=files, artifacts=artifacts,
             note="joint antenna x subcarrier MLPs vs joint LMMSE on validation run 0056 (odd blocks); "
                  "GMM numbers are copied from experiments/03 (1 seed); test run 0057 untouched")
    shutil.rmtree(weight_dir, ignore_errors=True)    # save_run copied them to data/runs/ (about 75 MB)


def plot(nmse, ref, a, path, test_note):
    from style import COLORS, LABELS, MARKERS, PAGE_W, apply_style
    plt = apply_style("paper")
    fig, axes = plt.subplots(1, len(a.patterns), figsize=(PAGE_W, 2.6), squeeze=False)
    shift = {v: (i - (len(a.variants) - 1) / 2) * 0.6 for i, v in enumerate(a.variants)}   # keep CI bars apart
    for ax, pname in zip(axes[0], a.patterns):
        ax.axhline(0, color=COLORS["lmmse"], linewidth=1.0, label="Joint " + LABELS["lmmse"] + " (reference)")
        g = [ref[pname][str(s)]["gmm_gain_db"] for s in SNRS]
        ax.plot(SNRS, g, color=COLORS["gmm"], linestyle="--", marker=MARKERS["gmm"], label="GMM (experiment 03, 1 seed)")
        for v in a.variants:
            e = [nmse[pname][str(s)][v]["ensemble"]["gain_over_lmmse_db"] for s in SNRS]
            y = np.array([x["value"] for x in e])
            err = [np.maximum(y - np.array([x["ci95"][0] for x in e]), 0),    # rounding can give -0.001
                   np.maximum(np.array([x["ci95"][1] for x in e]) - y, 0)]
            ax.errorbar(np.array(SNRS) + shift[v], y, yerr=err, color=COLORS[STYLE_KEY[v]], marker=MARKERS[STYLE_KEY[v]],
                        capsize=2, elinewidth=0.8,
                        label=f"{LEGEND[v]} ({len(a.seeds)} seed{'s' if len(a.seeds) > 1 else ''})")
        ants, sc = pname.split(" x ")
        n_pil = int(ants[:-3]) * int(sc[:-2])
        ax.set_title(f"Pilots: {ants[:-3]} antennas x {sc[:-2]} subcarriers ({n_pil} of 512)", loc="left")
        ax.set_xlabel("Pilot SNR (dB)")
        ax.set_xticks(SNRS)
    axes[0][0].set_ylabel("Gain over joint LMMSE (dB)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, -0.02))
    if test_note:
        fig.suptitle(f"CODE TEST, NOT A RESULT: {test_note}", fontsize=7, color="#B00020")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
