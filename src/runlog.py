"""Save one experiment run under results/ with the metadata needed to reproduce it.

    from runlog import save_run
    save_run("lmmse_baseline", "zhangsan",
             config={"block": [256, 320], "n_pilots": 8, "snr_db": [0, 10, 20, 30]},
             metrics={"nmse_db": {"0": -9.1, "10": -17.9}},
             files=["nmse_vs_snr.png"], note="first full-data run")

Creates results/<YYYYMMDD-HHMM>_<member>_<experiment>/ containing run.json and copies of
the given files. Large files are refused so the repository stays small.
"""
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_FILE_MB = 5


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_run(experiment, member, config, metrics, files=(), note=""):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out = os.path.join(ROOT, "results", f"{stamp}_{member}_{experiment}")
    os.makedirs(out, exist_ok=False)
    # only modified tracked files count; outputs a run just created (untracked) do not
    status = _git("status", "--porcelain", "--untracked-files=no")
    try:
        import numpy
        np_version = numpy.__version__
    except ImportError:
        np_version = None
    record = {
        "experiment": experiment,
        "member": member,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_uncommitted_changes": None if status is None else bool(status),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "numpy": np_version,
        "platform": platform.platform(),
        "note": note,
        "config": config,
        "metrics": metrics,
    }
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False, default=str)
    for p in files:
        mb = os.path.getsize(p) / 1e6
        if mb > MAX_FILE_MB:
            print(f"runlog: not copying {p} ({mb:.1f} MB > {MAX_FILE_MB} MB); keep it outside git")
            continue
        shutil.copy2(p, out)
    if record["git_uncommitted_changes"]:
        print("runlog: warning - code has uncommitted changes; commit before reporting this run")
    print("runlog: saved", out)
    return out
