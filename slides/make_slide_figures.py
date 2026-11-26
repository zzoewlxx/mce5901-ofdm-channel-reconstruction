"""Figures for the talk, drawn from the single test-run record (experiment 08, run 0057).

usage: python -I make_slide_figures.py <repo> <results/..._test-0057> <out dir>
"""
import json
import os
import sys

repo, rec_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(repo, "src"))
from style import COLORS, LABELS, MARKERS, apply_style  # noqa: E402

plt = apply_style("slides")
BG, INK, MUTED = "#FAF9F5", "#16233B", "#5B6475"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": MUTED, "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14, "savefig.dpi": 200})
with open(os.path.join(rec_dir, "run.json"), encoding="utf-8") as f:
    R = json.load(f)
assert R["config"]["run"] == 57 and not R["config"]["odd_only"]
M = R["metrics"]
SNRS = [0, 5, 10, 15, 20, 25, 30]
DENS = ["16", "8", "4"]
os.makedirs(out, exist_ok=True)


def st(k, **kw):
    return dict(color=COLORS[k], marker=MARKERS[k], label=LABELS[k], **kw)


def legend_below(fig, ax, ncol, y=-0.02):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


# ---- 1. NMSE vs SNR
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), sharey=True)
for ax, d in zip(axes, DENS):
    for k in ("linear", "spline", "lmmse"):
        ax.plot(SNRS, [M["main"][d][str(s)][k]["nmse_db"] for s in SNRS], **st(k))
    ax.plot(SNRS, [M["main"][d][str(s)]["mlp@mean"]["nmse_db"] for s in SNRS], **st("mlp", linestyle="--"))
    ax.set_title(f"{d} of 64 pilots", loc="left")
    ax.set_xlabel("Pilot SNR (dB)")
    ax.set_xticks([0, 10, 20, 30])
axes[0].set_ylabel("NMSE (dB)")
legend_below(fig, axes[0], 4)
fig.tight_layout()
fig.savefig(os.path.join(out, "s_nmse.png"))
plt.close(fig)

# ---- 2. gain over LMMSE
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), sharey=True)
for ax, d in zip(axes, DENS):
    ax.axhline(0, color=COLORS["lmmse"], linewidth=2.2, label="LMMSE (reference)")
    for k in ("mlp", "mlp_relweight", "residual", "mlp_blind"):
        g = [M["main"][d][str(s)][f"{k}@mean_gain_over_lmmse_db"] for s in SNRS]
        v = [x["value"] for x in g]
        lo = [max(x["value"] - x["ci95"][0], 0) for x in g]
        hi = [max(x["ci95"][1] - x["value"], 0) for x in g]
        ax.errorbar(SNRS, v, yerr=[lo, hi], capsize=3, elinewidth=1.2, **st(k))
    ax.set_title(f"{d} of 64 pilots", loc="left")
    ax.set_xlabel("Pilot SNR (dB)")
    ax.set_xticks([0, 10, 20, 30])
axes[0].set_ylabel("Gain over LMMSE (dB)")
legend_below(fig, axes[0], 3)
fig.tight_layout()
fig.savefig(os.path.join(out, "s_gain.png"))
plt.close(fig)

# ---- 3. mismatch (16 pilots)
mm = M["mismatch"]["16"]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.3))
off = mm["offset_loss_db"]
for k in ("lmmse", "mlp"):
    a.plot(off["offsets_db"], off["vs_own_matched"][k], **st(k))
a.set_title("Receiver assumes a wrong SNR", loc="left")
a.set_xlabel("Assumed − true SNR (dB)")
a.set_ylabel("NMSE loss (dB)")
a.set_xticks(range(-30, 31, 10))
fd = mm["fixed_design"]["20"]
ref = fd["lmmse_matched"]
b.axhline(0, color=COLORS["lmmse"], linewidth=1.6, linestyle=":", label="LMMSE, true SNR")
b.plot(SNRS, [x - y for x, y in zip(fd["lmmse"], ref)], color=COLORS["lmmse"], marker=MARKERS["lmmse"],
       label="LMMSE for 20 dB")
b.plot(SNRS, [x - y for x, y in zip(fd["mlp_blind"], ref)], **st("mlp_blind"))
b.set_title("Design for 20 dB vs. no SNR input", loc="left")
b.set_xlabel("True SNR (dB)")
b.set_ylabel("NMSE loss (dB)")
b.set_xticks([0, 10, 20, 30])
a.legend(frameon=False, loc="upper center")
b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.56, 1.02))
fig.tight_layout()
fig.savefig(os.path.join(out, "s_mismatch.png"))
plt.close(fig)

# ---- 4. joint setting: gain over joint LMMSE
JS = [0, 10, 20, 30]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
TIT = {"8ant x 4sc": "Pilots on 8 of 32 antennas", "32ant x 4sc": "Pilots on all 32 antennas"}
for ax, p in zip(axes, ("8ant x 4sc", "32ant x 4sc")):
    ax.axhline(0, color=COLORS["lmmse"], linewidth=2.2, label="Joint LMMSE")
    for k, key, lab in (("gmm", "gmm@mean", "GMM"), ("mlp", "jmlp@mean", "Joint MLP"),
                        ("residual", "jresidual@mean", "Residual MLP")):
        g = [M["joint"][p][str(s)][f"{key}_gain_over_lmmse_db"] for s in JS]
        v = [x["value"] for x in g]
        lo = [max(x["value"] - x["ci95"][0], 0) for x in g]
        hi = [max(x["ci95"][1] - x["value"], 0) for x in g]
        ax.errorbar(JS, v, yerr=[lo, hi], capsize=3, elinewidth=1.2, color=COLORS[k], marker=MARKERS[k],
                    markersize=11 if k == "gmm" else 7, label=lab)
    ax.set_title(TIT[p], loc="left")
    ax.set_xlabel("Pilot SNR (dB)")
    ax.set_xticks(JS)
axes[0].set_ylabel("Gain over LMMSE (dB)")
axes[0].set_ylim(-0.5, 6.8)
axes[1].set_ylim(-4.6, 2.8)
v30 = M["joint"]["32ant x 4sc"]["30"]["jmlp@mean_gain_over_lmmse_db"]["value"]
axes[1].annotate("joint MLP at 30 dB:" + chr(10) + f"{v30:.1f} dB".replace("-", "−"), xy=(21.6, -4.45),
                 xytext=(0.5, -4.1), color=COLORS["mlp"], fontsize=13,
                 arrowprops=dict(arrowstyle="->", color=COLORS["mlp"]))
legend_below(fig, axes[0], 4)
fig.tight_layout()
fig.savefig(os.path.join(out, "s_joint.png"))
plt.close(fig)
print("written", sorted(os.listdir(out)))
