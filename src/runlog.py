"""Save one experiment run under results/ with the metadata needed to reproduce it.

    from runlog import save_run
    save_run("lmmse_baseline", "zhangsan",
             config={"block": [256, 320], "n_pilots": 8, "snr_db": [0, 10, 20, 30]},
             metrics={"nmse_db": {"0": -9.1, "10": -17.9}},
             files=["nmse_vs_snr.png"], note="first full-data run",
             artifacts=["8_mlp.npz"])

Creates results/<YYYYMMDD-HHMM>_<member>_<experiment>/ containing run.json and copies of
the given files. Large files are refused so the repository stays small. `artifacts` (trained
weights and other large outputs) are copied to the git-ignored data/runs/<same folder name>/ and
their SHA-256 is recorded in run.json, so a later evaluation can prove which weights it used.
"""
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_FILE_MB = 5
CODE_DIRS = ("src", "experiments", "tools")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_state():
    """(uncommitted, untracked_code): modified tracked files, or Python files not yet committed.

    Outputs a run creates (untracked, not .py) do not count; a new script that was never
    committed does, because the recorded commit would not contain it.
    """
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *CODE_DIRS)
    if tracked is None or untracked is None:
        return None, []
    new_code = [f for f in untracked.splitlines() if f.endswith(".py")]
    return bool(tracked) or bool(new_code), new_code


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_run(experiment, member, config, metrics, files=(), note="", artifacts=()):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    name = f"{stamp}_{member}_{experiment}"
    out = os.path.join(ROOT, "results", name)
    os.makedirs(out, exist_ok=False)
    uncommitted, new_code = code_state()
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
        "git_uncommitted_changes": uncommitted,
        "untracked_code": new_code,
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "numpy": np_version,
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "platform": platform.platform(),
        "note": note,
        "config": config,
        "metrics": metrics,
    }
    if artifacts:
        art_dir = os.path.join(ROOT, "data", "runs", name)
        os.makedirs(art_dir, exist_ok=False)
        record["artifacts_dir"] = f"data/runs/{name}"
        record["artifacts"] = {}
        for p in artifacts:
            base = os.path.basename(p)
            shutil.copy2(p, os.path.join(art_dir, base))
            record["artifacts"][base] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False, default=str)
    for p in files:
        mb = os.path.getsize(p) / 1e6
        if mb > MAX_FILE_MB:
            print(f"runlog: not copying {p} ({mb:.1f} MB > {MAX_FILE_MB} MB); keep it outside git")
            continue
        shutil.copy2(p, out)
    if record["git_uncommitted_changes"]:
        print("runlog: warning - code has uncommitted changes" + (f" (new files {new_code})" if new_code else "")
              + "; commit before reporting this run")
    print("runlog: saved", out)
    return out


def load_artifacts(run_dir):
    """run.json of a saved run plus {file name: local path}, after checking every SHA-256."""
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        rec = json.load(f)
    art_dir = os.path.join(ROOT, rec["artifacts_dir"])
    paths = {}
    for base, info in rec["artifacts"].items():
        p = os.path.join(art_dir, base)
        if sha256(p) != info["sha256"]:
            raise ValueError(f"{p} does not match the SHA-256 recorded in {run_dir}/run.json")
        paths[base] = p
    return rec, paths
