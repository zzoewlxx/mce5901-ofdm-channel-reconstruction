"""Figures for the Chinese white paper (docs/whitepaper.md), drawn from the single test-run record.

usage: python -I docs/whitepaper_fig/make_figures.py <repo> <results/..._test-0057> <out dir>

Labels are Chinese, so a CJK font must be installed (Microsoft YaHei on Windows, PingFang SC on macOS,
Noto Sans CJK SC on Linux). Colours come from src/style.py, as in the report and the slides.
"""
import json
import os
import sys

repo, rec_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(repo, "src"))
from style import COLORS, MARKERS, apply_style  # noqa: E402

plt = apply_style("slides")
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
    "font.family": "sans-serif", "axes.unicode_minus": False,
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "savefig.dpi": 160,
})
with open(os.path.join(rec_dir, "run.json"), encoding="utf-8") as f:
    R = json.load(f)
assert R["config"]["run"] == 57 and not R["config"]["odd_only"], "expects the full test-run record of run 0057"
M = R["metrics"]
SNRS = [0, 5, 10, 15, 20, 25, 30]
DENS = ["16", "8", "4"]
os.makedirs(out, exist_ok=True)
NAMES = {"linear": "线性插值", "spline": "三次样条", "lmmse": "LMMSE", "mlp": "普通 MLP",
         "mlp_relweight": "相对误差损失", "residual": "残差 MLP（与 LMMSE 重合）", "mlp_blind": "盲 MLP（不需要 SNR）"}


def save(fig, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


# ---- 1. pilot layout
fig, ax = plt.subplots(figsize=(10, 2.6))
for row, n in enumerate([16, 8, 4]):
    pil = {round(i * 63 / (n - 1)) for i in range(n)}
    y = 2 - row
    for k in range(64):
        ax.add_patch(plt.Rectangle((k, y), 0.82, 0.7, color=COLORS["residual"] if k in pil else "#E4E3DF"))
    ax.text(-1.5, y + 0.35, f"{n} 个导频", ha="right", va="center", fontsize=12, fontweight="bold")
ax.set_xlim(-12, 64.5)
ax.set_ylim(-0.2, 3)
ax.axis("off")
ax.text(0.4, -0.15, "0", ha="center", va="top", fontsize=10, color="#5B6475")
ax.text(63.4, -0.15, "63", ha="center", va="top", fontsize=10, color="#5B6475")
ax.text(32, -0.15, "子载波序号（原始第 256–319 号，约 3.1 MHz）；蓝 = 导频（带噪声），灰 = 要推断",
        ha="center", va="top", fontsize=10, color="#5B6475")
save(fig, "fig_pilots.png")

# ---- 2. NMSE vs SNR
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
for ax, d in zip(axes, DENS):
    cell = M["main"][d]
    for k in ("linear", "spline", "lmmse"):
        ax.plot(SNRS, [cell[str(s)][k]["nmse_db"] for s in SNRS], color=COLORS[k], marker=MARKERS[k], label=NAMES[k])
    ax.plot(SNRS, [cell[str(s)]["mlp@mean"]["nmse_db"] for s in SNRS], color=COLORS["mlp"], marker=MARKERS["mlp"],
            linestyle="--", label="普通 MLP（3 种子平均）")
    gap = cell["0"]["linear"]["nmse_db"] - cell["0"]["lmmse"]["nmse_db"]
    ax.text(6, -5, f"0 dB 时 LMMSE 好 {gap:.1f} dB", fontsize=11)
    ax.set_title(f"{d} 个导频", loc="left", fontweight="bold")
    ax.set_xlabel("导频 SNR（dB）")
    ax.set_xticks([0, 10, 20, 30])
axes[0].set_ylabel("NMSE（dB，越低越好）")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
fig.tight_layout()
save(fig, "fig_nmse.png")

# ---- 3. gain of the MLP variants over LMMSE
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
for ax, d in zip(axes, DENS):
    ax.axhspan(-0.1, 0.1, color="#E4E3DF", alpha=0.8, linewidth=0)
    ax.axhline(0, color=COLORS["lmmse"], linewidth=2.2, label="LMMSE（参照 = 0）")
    for k in ("mlp", "mlp_relweight", "residual", "mlp_blind"):
        g = [M["main"][d][str(s)][f"{k}@mean_gain_over_lmmse_db"] for s in SNRS]
        v = [x["value"] for x in g]
        err = [[max(x["value"] - x["ci95"][0], 0) for x in g], [max(x["ci95"][1] - x["value"], 0) for x in g]]
        ax.errorbar(SNRS, v, yerr=err, color=COLORS[k], marker=MARKERS[k], capsize=2, label=NAMES[k],
                    linestyle="--" if k == "mlp_blind" else "-")
    ax.set_title(f"{d} 个导频", loc="left", fontweight="bold")
    ax.set_xlabel("导频 SNR（dB）")
    ax.set_xticks([0, 10, 20, 30])
axes[0].set_ylabel("相对 LMMSE 的增益（dB）")
axes[0].text(0, -1.72, "灰带 = ±0.1 dB，视为持平", fontsize=10, color="#5B6475")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, fontsize=10)
fig.tight_layout()
save(fig, "fig_gain.png")

# ---- 4. SNR mismatch, 16 pilots
mm = M["mismatch"]["16"]
off = mm["offset_loss_db"]
fx = mm["fixed_design"]["20"]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
for k in ("lmmse", "mlp"):
    a.plot(off["offsets_db"], off["vs_own_matched"][k], color=COLORS[k], marker=MARKERS[k],
           label=NAMES[k] if k == "lmmse" else "带 SNR 输入的普通 MLP")
a.axvline(0, color="#999999", linestyle=":", linewidth=1)
lo = off["vs_own_matched"]["lmmse"][off["offsets_db"].index(-10)]
hi = off["vs_own_matched"]["lmmse"][off["offsets_db"].index(10)]
a.text(2, 12, f"LMMSE：估低 10 dB 损失 {lo:.1f} dB\n而估高 10 dB 只损失 {hi:.1f} dB", fontsize=11)
a.set_title("(a) SNR 估偏多少，损失多少", loc="left", fontweight="bold")
a.set_xlabel("假设 SNR − 真实 SNR（dB）")
a.set_ylabel("相对自己用对 SNR 时的损失（dB）")
a.legend(frameon=False, loc="upper center")
base = fx["lmmse_matched"]
for k, name in (("lmmse", "LMMSE（按 20 dB 设计）"), ("mlp", "普通 MLP（输入 20 dB）"), ("mlp_blind", NAMES["mlp_blind"])):
    b.plot(SNRS, [x - y for x, y in zip(fx[k], base)], color=COLORS[k], marker=MARKERS[k], label=name,
           linestyle="--" if k == "mlp_blind" else "-")
b.set_title("(b) 一律按 20 dB 设计 vs 盲 MLP", loc="left", fontweight="bold")
b.set_xlabel("真实 SNR（dB）")
b.set_ylabel("相对知道真实 SNR 的 LMMSE（dB）")
b.set_xticks([0, 10, 20, 30])
b.legend(frameon=False, loc="upper center")
fig.tight_layout()
save(fig, "fig_mismatch.png")

# ---- 5. joint antenna x subcarrier setting
PATS = [("32ant x 4sc", "32 根天线 × 4 个子载波有导频（128 个）"), ("8ant x 4sc", "只有 8 根天线 × 4 个子载波有导频（32 个）")]
JS = ["0", "10", "20", "30"]
JM = [("gmm", "GMM", COLORS["gmm"]), ("jmlp", "联合 MLP", COLORS["mlp"]), ("jresidual", "联合残差 MLP", COLORS["residual"])]
LOW = -4
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, (p, title) in zip(axes, PATS):
    for j, (k, name, c) in enumerate(JM):
        for i, s in enumerate(JS):
            g = M["joint"][p][s][f"{k}@mean_gain_over_lmmse_db"]["value"]
            x = i + (j - 1) * 0.26
            ax.bar(x, max(g, LOW), width=0.24, color=c, alpha=0.9 if k == "gmm" else 0.6, label=name if i == 0 else None)
            if k == "gmm":
                ax.text(x, g + 0.15, f"+{g:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            if g < LOW:
                ax.text(x, LOW + 0.2, f"{g:.1f}↓", ha="center", va="bottom", fontsize=9.5, rotation=90)
    ax.axhline(0, color=COLORS["lmmse"], linewidth=2, label="联合 LMMSE（参照 = 0）")
    ax.set_xticks(range(len(JS)))
    ax.set_xticklabels([f"{s} dB\nLMMSE {M['joint'][p][s]['lmmse']['nmse_db']:.1f}" for s in JS], fontsize=10)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
    ax.set_ylim(LOW, 6.8)
axes[0].set_ylabel("相对联合 LMMSE 的增益（dB）")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
fig.tight_layout()
save(fig, "fig_joint.png")
print("wrote", sorted(f for f in os.listdir(out) if f.endswith(".png")))
