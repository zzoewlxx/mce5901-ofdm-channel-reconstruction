"""Load the processed block and build the fixed splits used by all experiments.

    from dataset import load_split
    d = load_split()                       # default npz, runs 53-55 fit, run 56 validation
    d.Hf, d.Hv                             # complex128 (n_vectors, 64), globally scaled
    d.sel, d.rep                           # masks on Hv: even blocks (selection) / odd blocks (report)
    d.rep_gid, d.n_rep_blocks              # block id of every reported vector (for bootstrap)

Protocol: vectors with estimated CSI error above `threshold` dB are removed; one global scale
from the fitting data; the validation run is cut into blocks of `block_len` consecutive
snapshots, even blocks for choosing hyper-parameters / early stopping, odd blocks for reporting.

The test run (0057) is refused by load_split and split_masks. It is returned only by
load_test(..., confirm=True), which the frozen test script calls once, after the freeze.
"""
import os
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NPZ = os.path.join(ROOT, "data", "processed", "dichasus005x_sc256-319.npz")
TEST_RUN = 57


def check_runs(fit_runs, val_run):
    """Refuse any split that would fit on, or select with, the held-out test run."""
    if TEST_RUN in tuple(fit_runs) or val_run == TEST_RUN:
        raise ValueError(f"run {TEST_RUN:04d} is the held-out test set: only load_test(..., confirm=True), "
                         "called once by the frozen test script, may load it")
    if val_run in fit_runs:
        raise ValueError("validation run must not be a fitting run")


def split_masks(run, index, fit_runs=(53, 54, 55), val_run=56, smoke=False):
    """Snapshot masks (fit, validation).

    smoke: a quick code test on ONE fitting run (its first 70% of snapshots fit, the rest validate),
    so it never touches the validation or the test run, whatever file is given.
    """
    check_runs(fit_runs, val_run)
    if not smoke:
        return np.isin(run, fit_runs), run == val_run
    base = run == fit_runs[0]
    if not base.any():                                  # a single-run file of another run
        present = np.unique(run)
        if len(present) != 1 or present[0] == TEST_RUN:
            raise ValueError(f"smoke mode needs run {fit_runs[0]} or a single non-test run, file has {present}")
        base = run == present[0]
    cut = np.quantile(index[base], 0.7)
    return base & (index < cut), base & (index >= cut)


def _vectors(csi, keep, snap_mask):
    m = snap_mask[:, None] & keep
    snap_id = np.broadcast_to(np.arange(len(keep))[:, None], keep.shape)[m]
    return csi[m].astype(np.complex128), snap_id


def load_split(path=DEFAULT_NPZ, fit_runs=(53, 54, 55), val_run=56, threshold=-30.0,
               block_len=50, smoke=False):
    z = np.load(path)
    csi, quality, run, index = z["csi"], z["quality"], z["run"], z["index"]
    fit_snap, val_snap = split_masks(run, index, fit_runs, val_run, smoke)
    keep = quality <= threshold
    Hf, _ = _vectors(csi, keep, fit_snap)
    Hv, sid = _vectors(csi, keep, val_snap)
    block = index[sid] // block_len
    sel, rep = block % 2 == 0, block % 2 == 1
    rep_blocks, rep_gid = np.unique(block[rep], return_inverse=True)
    scale = 1 / np.sqrt(np.mean(np.abs(Hf) ** 2))
    return SimpleNamespace(Hf=Hf * scale, Hv=Hv * scale, sel=sel, rep=rep, rep_gid=rep_gid,
                           n_rep_blocks=len(rep_blocks), scale=float(scale), n=csi.shape[2],
                           block=z["block"].tolist(), threshold=threshold, block_len=block_len)


def load_test(path=DEFAULT_NPZ, fit_runs=(53, 54, 55), test_run=TEST_RUN, threshold=-30.0,
              block_len=50, confirm=False):
    """Fitting data plus the WHOLE evaluation run, scaled with the fitting-data scale.

    Nothing may be selected on the returned run: every hyper-parameter must already be frozen.
    Run 0057 needs confirm=True. test_run=56 rehearses the same code path on the validation run;
    d.odd then marks the odd (reported) blocks, so the rehearsal can reproduce the validation numbers.
    """
    if test_run == TEST_RUN and not confirm:
        raise PermissionError(f"loading test run {TEST_RUN:04d} needs confirm=True (once, after the freeze)")
    if test_run in tuple(fit_runs) or TEST_RUN in tuple(fit_runs):
        raise ValueError("the evaluation run must not be a fitting run, and the test run never is one")
    z = np.load(path)
    csi, quality, run, index = z["csi"], z["quality"], z["run"], z["index"]
    keep = quality <= threshold
    Hf, _ = _vectors(csi, keep, np.isin(run, fit_runs))
    Ht, sid = _vectors(csi, keep, run == test_run)
    block = index[sid] // block_len
    blocks, gid = np.unique(block, return_inverse=True)
    scale = 1 / np.sqrt(np.mean(np.abs(Hf) ** 2))
    return SimpleNamespace(Hf=Hf * scale, Ht=Ht * scale, gid=gid, n_blocks=len(blocks), odd=block % 2 == 1,
                           block_of=block, scale=float(scale), n=csi.shape[2], block=z["block"].tolist(),
                           threshold=threshold, block_len=block_len, test_run=test_run,
                           screened_fraction=round(float(np.mean(quality[run == test_run] > threshold)), 5))


def bootstrap_ci(err_by_block, ref_by_block, boots):
    """95% CI of NMSE (dB) from per-block sums; boots = (B, n_blocks) resampled block indices."""
    v = 10 * np.log10(err_by_block[boots].sum(1) / ref_by_block[boots].sum(1))
    return v, [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
