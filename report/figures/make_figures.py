"""Figures of the report, drawn from saved run records only (no data file, no recomputation).

    python report/figures/make_figures.py --main <run folder of experiment 08> --mismatch <run folder of 04>

The suffix of each figure name says which run the numbers come from: "_test" for test run 0057 (the record
of THE test), "_val" for validation run 0056 (a rehearsal record, or experiment 04). main.tex includes the
"_test" files, so a figure drawn from validation numbers can never end up in the report by accident.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from style import COL_W, COLORS, LABELS, MARKERS, PAGE_W, apply_style  # noqa: E402

plt = apply_style("paper")
DENS = ("16", "8", "4")
VARIANTS = ("mlp", "mlp_relweight", "residual", "mlp_blind")


def record(folder):
    path = folder if folder.endswith(".json") else os.path.join(folder, "run.json")
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tag_of(rec):
    run = rec["config"].get("run", 56)
    return "test" if run == 57 else "val"


def style(key, **kw):
    return dict(color=COLORS[key], marker=MARKERS[key], label=LABELS[key], **kw)


def fig_nmse(main, path):
    """NMSE versus pilot SNR, one panel per pilot count (Fig. 1, full page width)."""
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.95), sharey=True)
    for ax, dens in zip(axes, DENS):
        rows = main[dens]
        snrs = sorted(map(int, rows))
        for key, src in (("linear", "linear"), ("spline", "spline"), ("lmmse", "lmmse"), ("mlp", "mlp@mean"),
                         ("mlp_blind", "mlp_blind@mean")):
            ax.plot(snrs, [rows[str(s)][src]["nmse_db"] for s in snrs], **style(key))
        ax.set_title(f"{dens} of 64 subcarriers are pilots", loc="left")
        ax.set_xlabel("Pilot SNR (dB)")
        ax.set_xticks(snrs[::2] if len(snrs) > 4 else snrs)
    axes[0].set_ylabel("NMSE (dB)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    worst = max(max(abs(rows[str(s)][k]["ci95"][1] - rows[str(s)][k]["ci95"][0]) / 2 for s in map(int, rows)
                    for k in ("linear", "spline", "lmmse", "mlp@mean", "mlp_blind@mean")) for rows in main.values())
    return {"max_ci_half_width_db": round(worst, 2)}


def fig_gain(main, path):
    """Gain of each learned estimator over LMMSE with its paired 95% CI (Fig. 2, one column)."""
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 2.1), sharey=True)
    off = {"mlp": -0.9, "mlp_relweight": -0.3, "residual": 0.3, "mlp_blind": 0.9}   # dodge overlapping bars
    for ax, dens in zip(axes, DENS):
        rows = main[dens]
        snrs = np.array(sorted(map(int, rows)))
        ax.axhline(0, color=COLORS["lmmse"], lw=1.0)
        for v in VARIANTS:
            g = [rows[str(s)][f"{v}@mean_gain_over_lmmse_db"] for s in snrs]
            val = np.array([x["value"] for x in g])
            lo, hi = np.array([x["ci95"][0] for x in g]), np.array([x["ci95"][1] for x in g])
            yerr = [np.maximum(val - lo, 0), np.maximum(hi - val, 0)]   # 3-decimal records: value may sit on an end
            ax.errorbar(snrs + off[v], val, yerr=yerr, capsize=1.2, elinewidth=0.7,
                        lw=1.0, ms=3, **style(v))
        ax.set_title(f"{dens} pilots", loc="left")
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Gain over LMMSE (dB)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def mismatch_summary(rec, dens):
    """Experiment 04's summary of one pilot count, from a 04 record (validation) or an 08 record (test)."""
    m = rec["metrics"]
    return m["mismatch"][dens] if "mismatch" in m else {k: m[k][dens] for k in ("offset_loss_db", "fixed_design")}


def fig_mismatch(rec, path, dens="16"):
    """(a) loss versus assumed - true SNR; (b) receivers designed for 20 dB, loss versus the true SNR
    relative to the LMMSE that knows it (Fig. 3, one column)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.9))
    mm = mismatch_summary(rec, dens)
    ol = mm["offset_loss_db"]
    for v in ("lmmse", "mlp", "mlp_relweight"):
        a.plot(ol["offsets_db"], ol["vs_own_matched"][v], lw=1.0, ms=3, **style(v))
    a.set_xlabel("Assumed $-$ true SNR (dB)")
    a.set_ylabel("NMSE loss (dB)")
    a.set_xticks([-30, -15, 0, 15, 30])
    a.set_title("(a) wrong SNR", loc="left")
    fd = mm["fixed_design"]["20"]
    snrs = [0, 5, 10, 15, 20, 25, 30]                     # experiment 04's grid (true SNR)
    ref = np.array(fd["lmmse_matched"])
    if len(ref) != len(snrs):
        raise ValueError("experiment 04 record does not use the 0:5:30 dB grid")
    b.axhline(0, color=COLORS["genie"], lw=1.0, ls="--", label="LMMSE, true SNR known")
    for v in ("lmmse", "mlp", "mlp_blind"):
        b.plot(snrs, np.array(fd[v]) - ref, lw=1.0, ms=3, **style(v))
    b.axvline(20, color="#888888", lw=0.6, ls=":")
    b.set_xlabel("True SNR (dB)")
    b.set_ylabel("NMSE loss (dB)")
    b.set_xticks([0, 10, 20, 30])
    b.set_title("(b) designed for 20 dB", loc="left")
    h, lab = a.get_legend_handles_labels()
    h2, lab2 = b.get_legend_handles_labels()
    h += [x for x, l in zip(h2, lab2) if l not in lab]
    lab += [l for l in lab2 if l not in lab]
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.24, 1, 1), w_pad=0.8)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", help="run folder (or run.json) of experiment 08: rehearsal or the test")
    ap.add_argument("--mismatch", help="run folder (or run.json) of experiment 04 (validation version of Fig. 3)")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    done = []
    if a.main:
        rec = record(a.main)
        t = tag_of(rec)
        info = fig_nmse(rec["metrics"]["main"], os.path.join(a.out, f"fig_nmse_{t}.{a.fmt}"))
        fig_gain(rec["metrics"]["main"], os.path.join(a.out, f"fig_gain_{t}.{a.fmt}"))
        done += [f"fig_nmse_{t}.{a.fmt} ({info})", f"fig_gain_{t}.{a.fmt}"]
        if "mismatch" in rec["metrics"]:
            fig_mismatch(rec, os.path.join(a.out, f"fig_mismatch_{t}.{a.fmt}"))
            done.append(f"fig_mismatch_{t}.{a.fmt}")
    if a.mismatch:                                       # experiment 04's record: validation run 0056
        fig_mismatch(record(a.mismatch), os.path.join(a.out, f"fig_mismatch_val.{a.fmt}"))
        done.append(f"fig_mismatch_val.{a.fmt}")
    print("written to", a.out, ":", "; ".join(done))


if __name__ == "__main__":
    main()
