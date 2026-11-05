"""Download dichasus-005x (DOI 10.18419/darus-2204) with resume and MD5 check.

Usage:
    python tools/download_dichasus.py                      # all 5 runs (6.87 GB) -> data/raw/dichasus-005x
    python tools/download_dichasus.py --runs 0053 0057     # only some runs
    python tools/download_dichasus.py --dest D:/somewhere  # other folder

Interrupted downloads resume where they stopped; finished files are skipped
after their MD5 has been verified.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DARUS = "https://darus.uni-stuttgart.de/api/access/datafile/{}"
FILES = {  # run: (DaRUS file id, size in bytes, md5) - from the DaRUS dataset metadata
    "0053": (68398, 1496317592, "915cc51a7c82eff19200ac53dd277d34"),
    "0054": (68399, 1360288720, "c20230ab7e1fbe28b3fda6eb3ca1e881"),
    "0055": (68400, 1329301448, "02837b34236fa55b2b56020e69eb8a38"),
    "0056": (68401, 1370792880, "b15f469dfd78b7cba310693d91f6e7c9"),
    "0057": (68402, 1313282604, "3f043a8f4fa37e853cd45f087062f475"),
}
EXTRAS = {
    "spec.json": DARUS.format(68517),
    "reftx-offsets-dichasus-005x.json": "https://dichasus.inue.uni-stuttgart.de/datasets/data/"
                                        "dichasus-005x/reftx-offsets-dichasus-005x.json",
}


def md5sum(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, size=None, retries=20):
    for attempt in range(retries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if size is not None and have >= size:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                if have and r.status != 206:      # server ignored the range: restart
                    have = 0
                    open(path, "wb").close()
                t0, got, last = time.time(), 0, 0
                with open(path, "ab") as f:
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
                        got += len(block)
                        if size and (have + got) * 20 // size > last:
                            last = (have + got) * 20 // size
                            rate = got / 1e6 / max(time.time() - t0, 1e-6)
                            print(f"  {os.path.basename(path)}: {100 * (have + got) / size:5.1f}%  "
                                  f"{rate:5.2f} MB/s", flush=True)
            if size is None or os.path.getsize(path) >= size:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"  retry {attempt + 1}/{retries} after error: {e}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"failed to download {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(ROOT, "data", "raw", "dichasus-005x"))
    ap.add_argument("--runs", nargs="*", default=list(FILES))
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    print("destination:", a.dest, flush=True)

    for name, url in EXTRAS.items():
        p = os.path.join(a.dest, name)
        if not os.path.exists(p):
            fetch(url, p)
    ok = True
    for run in a.runs:
        fid, size, md5 = FILES[run]
        path = os.path.join(a.dest, f"dichasus-{run}.tfrecords")
        if os.path.exists(path) and os.path.getsize(path) == size and md5sum(path) == md5:
            print(f"dichasus-{run}: already complete, MD5 ok", flush=True)
            continue
        if os.path.exists(path) and os.path.getsize(path) > size:
            os.remove(path)
        print(f"dichasus-{run}: downloading {size / 1e9:.2f} GB", flush=True)
        fetch(DARUS.format(fid), path, size)
        good = md5sum(path) == md5
        ok &= good
        print(f"dichasus-{run}: {'MD5 ok' if good else 'MD5 MISMATCH - delete the file and rerun'}", flush=True)
    print("ALL DONE, all checksums ok" if ok else "FINISHED WITH ERRORS", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
