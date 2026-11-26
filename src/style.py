"""One palette and figure style for every plot in the report and the slides.

    from style import COLORS, MARKERS, LABELS, apply_style
    apply_style("paper")            # IEEE: 8 pt text; widths COL_W (one column) / PAGE_W (two)
    ax.plot(x, y, color=COLORS["lmmse"], marker=MARKERS["lmmse"], label=LABELS["lmmse"])

Colours follow the Okabe-Ito colour-blind-safe set; a method keeps its colour in every figure.
A method family keeps its colour across settings (the joint-setting MLP uses the MLP colour).
"""
COLORS = {
    "linear": "#56B4E9",          # sky blue
    "spline": "#E69F00",          # orange
    "lmmse": "#009E73",           # green: the reference every gain is measured against
    "mlp": "#D55E00",             # vermillion
    "mlp_relweight": "#CC79A7",   # reddish purple
    "residual": "#0072B2",        # blue
    "mlp_blind": "#4D4D4D",       # dark grey (#7F7F7F matched the LMMSE green under deuteranopia)
    "mlp_wide": "#8C564B",        # brown
    "gmm": "#000000",             # black
    "genie": "#999999",           # light grey, dashed: references that are not usable receivers
}
MARKERS = {"linear": "o", "spline": "s", "lmmse": "D", "mlp": "^", "mlp_relweight": "v", "residual": "P",
           "mlp_blind": "X", "mlp_wide": "<", "gmm": "*", "genie": ""}
LABELS = {"linear": "Linear interpolation", "spline": "Cubic spline", "lmmse": "LMMSE",
          "mlp": "MLP", "mlp_relweight": "MLP, relative-error loss", "residual": "LMMSE + residual MLP",
          "mlp_blind": "MLP, no SNR input", "mlp_wide": "MLP 3x256", "gmm": "GMM estimator",
          "genie": "Position genie (not a receiver)"}

COL_W, PAGE_W = 3.5, 7.16           # IEEE column / page width in inches
SLIDE_W = 11.0                      # figure width for a 16:9 slide, inches


def apply_style(target="paper"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = 8 if target == "paper" else 14
    plt.rcParams.update({
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1, "legend.fontsize": base - 1,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#E4E3DF", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.4 if target == "paper" else 2.2, "lines.markersize": 4 if target == "paper" else 7,
        "savefig.dpi": 300 if target == "paper" else 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt
