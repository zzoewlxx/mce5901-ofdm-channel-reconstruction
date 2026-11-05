"""Plot the pilot findings stored in pilot_results.json (written by pilot_check.py).

Usage:  python -I make_figure.py [pilot_results.json] [output.png]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pilot_results.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "pilot_findings.png")

SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
SERIES = {  # fixed categorical order: slot 1, 2, 3 (validated, light mode)
    "linear": ("Linear interpolation", "#2a78d6", "o"),
    "spline": ("Cubic spline", "#eb6834", "s"),
    "lmmse": ("LMMSE (training covariance)", "#1baf7a", "D"),
}
DF_MHZ = 50 / 1024

with open(SRC, encoding="utf-8") as f:
    R = json.load(f)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK2)


# (a) frequency correlation -------------------------------------------------
ax = axes[0]
lags = np.asarray(R["fcf"]["lags"])
corr = np.asarray(R["fcf"]["abs_corr"])
x = lags * DF_MHZ
ax.plot(x, corr, color=SERIES["linear"][1], linewidth=2)
for xv, txt, at in [(9 * DF_MHZ, "proposal pilot spacing\n(8 of 64 = every 9th, 0.44 MHz)", (3.8, 0.78)),
                    (64 * DF_MHZ, "whole 64-subcarrier block\n(3.1 MHz)", (6.5, 0.955))]:
    ax.axvline(xv, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    yv = float(np.interp(xv, x, corr))
    ax.plot([xv], [yv], "o", color=SERIES["linear"][1], markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.annotate(f"{txt}\n|R| = {yv:.3f}", (xv, yv), xytext=at, textcoords="data",
                fontsize=9, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.set_xlim(0, x.max())
ax.set_ylim(min(0.65, corr.min() - 0.02), 1.005)
ax.set_xlabel("Frequency separation Δf (MHz)")
ax.set_ylabel("|frequency correlation|")
ax.set_title("(a) The measured channel is very smooth in frequency")

# (b) noise-free pilots: spacing sweep ---------------------------------------
ax = axes[1]
wide = R["wide_cfg"]
sp = np.array([w["spacing_kHz"] for w in wide])
floor_ref = float(np.median([r["floor"] for r in R["proposal_cfg"]]))
ax.axhline(floor_ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(sp[0], floor_ref + 0.4, "measured-CSI noise level (delay-domain est.)",
        fontsize=8.5, color=INK2, va="bottom", ha="left")
for key, (name, col, mk) in SERIES.items():
    y = np.array([w[key] for w in wide])
    ax.plot(sp, y, color=col, linewidth=2, marker=mk, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
prop = [r for r in R["proposal_cfg"] if r["block_start"] == 256 and r["n_obs"] in (8, 16)]
lo = min(min(p["linear"], p["lmmse"]) for p in prop)
hi = max(max(p["linear"], p["lmmse"]) for p in prop)
ax.annotate(f"proposal setting (64-block, 8 or 16 pilots):\nall methods {hi:.0f} to {lo:.0f} dB",
            (391, prop[0]["linear"]), xytext=(230, -34.5), textcoords="data",
            fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.set_xscale("log")
ax.set_xticks(sp)
ax.set_xticklabels([f"{v:.0f}" for v in sp], fontsize=8)
ax.minorticks_off()
ax.set_xlabel("Pilot spacing (kHz), noise-free pilots, 448-subcarrier block")
ax.set_ylabel("NMSE on withheld subcarriers (dB)")
ax.set_title("(b) Noise-free pilots: ≤ −40 dB until spacing > 1.5 MHz")
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

# (c) noisy pilots ------------------------------------------------------------
ax = axes[2]
noisy = [r for r in R["noisy_cfg"] if r["cfg"] == "block64_obs8"]
snr = np.array([r["pilot_snr_db"] for r in noisy])
for key, (name, col, mk) in SERIES.items():
    y = np.array([r[key] for r in noisy])
    ax.plot(snr, y, color=col, linewidth=2, marker=mk, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    end_label(ax, snr[-1], y[-1], name.split(" (")[0], dy={"linear": -9, "spline": 9, "lmmse": 0}[key])
i10 = int(np.where(snr == 10)[0][0])
lin10, lm10 = noisy[i10]["linear"], noisy[i10]["lmmse"]
ax.annotate("", (10, lm10), xytext=(10, lin10),
            arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1))
ax.annotate(f"{lin10 - lm10:.1f} dB gap at 10 dB pilot SNR\n= room for a learned estimator",
            ((10, (lin10 + lm10) / 2)), xytext=(14, -5), textcoords="data",
            fontsize=9, color=INK2, va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.set_xlim(snr.min() - 1, snr.max() + 9)
ax.set_xlabel("Pilot SNR (dB): measured CSI + added complex Gaussian noise")
ax.set_ylabel("NMSE on withheld subcarriers (dB)")
ax.set_title("(c) Noisy pilots (64-block, 8 of 64): methods separate")
ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)

s = R["sample"]
fig.text(0.01, 0.01,
         f"Pilot study on a sample of dichasus-005x: fit on {s['train_snapshots']} snapshots of run 0053, "
         f"evaluated on {s['eval_snapshots']} snapshots of run 0056 (32 antennas pooled). "
         "Test run 0057 not used. Pilot numbers, not final results.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print("saved", OUT)
