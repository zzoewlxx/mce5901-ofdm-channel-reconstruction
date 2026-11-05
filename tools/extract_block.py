"""Extract one subcarrier block of dichasus-005x into a single compact .npz file.

For every snapshot and antenna it stores the calibrated CSI of the block plus a quality
measure (CSI error estimated from 4th-order frequency differences over the clean band,
in dB relative to the channel power), so screening can be applied later.

    python tools/extract_block.py                         # block 256-319, all five runs
    python tools/extract_block.py --start 384 --width 64  # another block
    python tools/extract_block.py --runs 0053             # quick test on one run

Output arrays (N snapshots, 32 antennas, W subcarriers):
    csi      complex64 (N, 32, W)  calibrated (REFTX offsets), fftshifted, block only
    quality  float32   (N, 32)     estimated CSI error in dB (screen e.g. quality > -30)
    snr      float32   (N, 32)     the dataset's own SNR field (NOT a reliable quality flag)
    pos      float64   (N, 2)      LiDAR position (m);  time, rot: float64 (N,)
    run      int16     (N,)        53..57;  index: int32 (N,) record index within the run
    block    int32     [start, width]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dichasus_reader import iter_file  # noqa: E402

RUNS = ["0053", "0054", "0055", "0056", "0057"]
_K = np.r_[np.arange(130, 508), np.arange(517, 892)]  # 4th-difference centres, clean band
_BAND = np.r_[128:510, 515:894]


def csi_error_db(H):
    """H: (..., 1024) -> estimated CSI error relative to in-band power, dB."""
    d4 = H[..., _K - 2] - 4 * H[..., _K - 1] + 6 * H[..., _K] - 4 * H[..., _K + 1] + H[..., _K + 2]
    s = np.mean(np.abs(d4) ** 2, axis=-1) / 70.0
    p = np.mean(np.abs(H[..., _BAND]) ** 2, axis=-1)
    return 10 * np.log10(s / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw", "dichasus-005x"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=int, default=256)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    a = ap.parse_args()
    sl = slice(a.start, a.start + a.width)
    if not (0 <= a.start and sl.stop <= 1024) or a.start <= 512 < sl.stop:
        print("warning: block touches the band edge or the DC subcarrier 512")
    out = a.out or os.path.join(ROOT, "data", "processed",
                                f"dichasus005x_sc{a.start}-{sl.stop - 1}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(os.path.join(a.raw, "reftx-offsets-dichasus-005x.json"), encoding="utf-8") as f:
        off = json.load(f)
    k = np.arange(1024)
    corr = np.exp(1j * (np.asarray(off["sto"])[:, None] * 2 * np.pi * k[None, :] / 1024
                        + np.asarray(off["cpo"])[:, None])).astype(np.complex64)

    cols = {n: [] for n in ("csi", "quality", "snr", "pos", "time", "rot", "run", "index")}
    t0 = time.time()
    for run in a.runs:
        path = os.path.join(a.raw, f"dichasus-{run}.tfrecords")
        n0 = len(cols["run"])
        for i, rec in enumerate(iter_file(path)):
            H = rec["csi"] * corr                      # (32, 1024) calibrated
            if not np.all(np.isfinite(H)):
                print(f"  {run} record {i}: non-finite values, skipped")
                continue
            cols["csi"].append(H[:, sl])
            cols["quality"].append(csi_error_db(H.astype(np.complex128)).astype(np.float32))
            cols["snr"].append(rec["snr"])
            cols["pos"].append(rec["pos"])
            cols["time"].append(rec["time"])
            cols["rot"].append(rec["rot"])
            cols["run"].append(int(run))
            cols["index"].append(i)
        print(f"  dichasus-{run}: {len(cols['run']) - n0} snapshots  ({time.time() - t0:.0f} s)", flush=True)

    arrays = {
        "csi": np.stack(cols["csi"]).astype(np.complex64),
        "quality": np.stack(cols["quality"]).astype(np.float32),
        "snr": np.stack(cols["snr"]).astype(np.float32),
        "pos": np.stack(cols["pos"]).astype(np.float64),
        "time": np.asarray(cols["time"], np.float64),
        "rot": np.asarray(cols["rot"], np.float64),
        "run": np.asarray(cols["run"], np.int16),
        "index": np.asarray(cols["index"], np.int32),
        "block": np.asarray([a.start, a.width], np.int32),
    }
    np.savez(out, **arrays)
    q = arrays["quality"]
    print(f"saved {out}: {arrays['csi'].shape}, {os.path.getsize(out) / 1e6:.0f} MB; "
          f"vectors with CSI error > -30 dB: {np.mean(q > -30) * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
