"""Unattended pipeline: download -> extract -> baselines -> push results -> data release.

    python tools/run_pipeline.py --member <name>

Steps (each one is skipped or resumed if already done):
  1. tools/download_dichasus.py         resume + MD5 check of the 5 runs
  2. tools/extract_block.py             data/processed/dichasus005x_sc256-319.npz
  3. experiments/01_baselines_fulldata  baselines on validation run 0056 -> results/
  4. git commit + push of the new results
  5. GitHub release "data-v1" with the processed .npz (non-fatal if the upload fails)
  6. comment on issue #4 with a link to the results (non-fatal)
Keeps Windows from idle-sleeping while it runs. Log: data/logs/pipeline_<time>.log
"""
import argparse
import datetime
import glob
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "data", "processed", "dichasus005x_sc256-319.npz")
RELEASE_TAG = "data-v1"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
LOG = None


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, check=True):
    log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    out = []
    for line in p.stdout:
        line = line.rstrip()
        out.append(line)
        log("  " + line)
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.returncode, "\n".join(out)


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--no-release", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "data", "logs"), exist_ok=True)
    LOG = os.path.join(ROOT, "data", "logs", f"pipeline_{datetime.datetime.now():%Y%m%d-%H%M}.log")
    keep_awake()
    py = [sys.executable, "-I"]

    for attempt in range(3):
        code, _ = run(py + ["tools/download_dichasus.py"], check=False)
        if code == 0:
            break
        log(f"download attempt {attempt + 1} failed, retrying")
    else:
        raise RuntimeError("download failed three times; see log")

    if not os.path.exists(NPZ):
        run(py + ["tools/extract_block.py"])
    else:
        log(f"{NPZ} exists, extraction skipped")

    before = set(glob.glob(os.path.join(ROOT, "results", "*")))
    run(py + ["experiments/01_baselines_fulldata/run_baselines.py", "--member", a.member])
    new = sorted(set(glob.glob(os.path.join(ROOT, "results", "*"))) - before)
    if not new:
        raise RuntimeError("baseline run produced no results folder")
    res_dir = os.path.relpath(new[-1], ROOT).replace(os.sep, "/")

    run(["git", "add", res_dir, "experiments/01_baselines_fulldata/nmse_vs_snr_val.png"])
    run(["git", "commit", "-m", f"Add full-data baseline results on validation run 0056 ({res_dir})" + TRAILER])
    run(["git", "pull", "--rebase"], check=False)
    run(["git", "push"])
    _, commit = run(["git", "rev-parse", "--short", "HEAD"])
    url = f"https://github.com/zzoewlxx/mce5901-ofdm-channel-reconstruction/tree/main/{res_dir}"

    if not a.no_release:
        code, _ = run(["gh", "release", "view", RELEASE_TAG], check=False)
        if code != 0:
            md5 = md5sum(NPZ)
            notes = (
                "Processed subset of the DICHASUS dataset dichasus-005x (F. Euchner, M. Gauger, DaRUS 2021, "
                "doi:10.18419/darus-2204, CC BY 4.0).\n\n"
                "- Subcarriers 256-319 after fftshift, all 32 antennas, all 5 runs (26,161 snapshots)\n"
                "- REFTX phase/timing offsets compensated; per-vector CSI quality in dB (screen with quality > -30)\n"
                f"- Created with tools/extract_block.py; repository commit {commit.strip()}\n"
                f"- MD5 {md5}\n\n"
                "Load: z = numpy.load('dichasus005x_sc256-319.npz'); z['csi'], z['quality'], z['run'], ...\n"
                "To build other blocks, download the raw data with tools/download_dichasus.py.")
            code, _ = run(["gh", "release", "create", RELEASE_TAG, NPZ, "--title",
                           "Processed data v1: dichasus-005x, subcarriers 256-319", "--notes", notes], check=False)
            log("release upload " + ("done" if code == 0 else "FAILED (rerun: gh release create ... later)"))
        else:
            log(f"release {RELEASE_TAG} already exists, skipped")

    body = (f"自动流程已在全量数据上跑完基线（验证集 run 0056，测试集 0057 未使用）：{url}\n\n"
            "处理后的数据（256–319 子载波块）见 Releases → data-v1。请组员核对后再关闭本 issue。")
    run(["gh", "issue", "comment", "4", "--body", body], check=False)
    log("PIPELINE FINISHED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # keep the reason in the log
        if LOG:
            log(f"PIPELINE FAILED: {e}")
        raise
