"""Load the processed block and build the fixed splits used by all experiments.

    from dataset import load_split
    d = load_split()                       # default npz, runs 53-55 fit, run 56 validation
    d.Hf, d.Hv                             # complex128 (n_vectors, 64), globally scaled
    d.sel, d.rep                           # masks on Hv: even blocks (selection) / odd blocks (report)
    d.rep_gid, d.n_rep_blocks              # block id of every reported vector (for bootstrap)

Protocol: vectors with estimated CSI error above `threshold` dB are removed; one global scale
from the fitting data; the validation run is cut into blocks of `block_len` consecutive
snapshots, even blocks for choosing hyper-parameters / early stopping, odd blocks for reporting.
The test run (0057) is never loaded here.
"""
import os
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NPZ = os.path.join(ROOT, "data", "processed", "dichasus005x_sc256-319.npz")


def load_split(path=DEFAULT_NPZ, fit_runs=(53, 54, 55), val_run=56, threshold=-30.0,
               block_len=50, smoke=False):
    z = np.load(path)
    csi, quality, run, index = z["csi"], z["quality"], z["run"], z["index"]
    if smoke:  # single-run file: first 70% fit, rest validation
        cut = np.quantile(index, 0.7)
        fit_snap, val_snap = index < cut, index >= cut
    else:
        if val_run in fit_runs:
            raise ValueError("validation run must not be a fitting run")
        fit_snap, val_snap = np.isin(run, fit_runs), run == val_run
    keep = quality <= threshold

    def vectors(snap_mask):
        m = snap_mask[:, None] & keep
        snap_id = np.broadcast_to(np.arange(len(run))[:, None], keep.shape)[m]
        return csi[m].astype(np.complex128), snap_id

    Hf, _ = vectors(fit_snap)
    Hv, sid = vectors(val_snap)
    block = index[sid] // block_len
    sel, rep = block % 2 == 0, block % 2 == 1
    rep_blocks, rep_gid = np.unique(block[rep], return_inverse=True)
    scale = 1 / np.sqrt(np.mean(np.abs(Hf) ** 2))
    return SimpleNamespace(Hf=Hf * scale, Hv=Hv * scale, sel=sel, rep=rep, rep_gid=rep_gid,
                           n_rep_blocks=len(rep_blocks), scale=float(scale), n=csi.shape[2],
                           block=z["block"].tolist(), threshold=threshold, block_len=block_len)


def bootstrap_ci(err_by_block, ref_by_block, boots):
    """95% CI of NMSE (dB) from per-block sums; boots = (B, n_blocks) resampled block indices."""
    v = 10 * np.log10(err_by_block[boots].sum(1) / ref_by_block[boots].sum(1))
    return v, [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
