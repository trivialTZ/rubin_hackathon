"""Generate the v7 calibrated-classifier AUC chart.

AUC values are SCC-derived constants from the v7 trust run.  The corresponding
SCC metrics JSON is not present in this local checkout, so this script keeps the
reviewed constants explicit while styling the chart deterministically.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from metadebass_plot_style import PALETTE, apply_neurips_style, clean_axis, save_figure

apply_neurips_style(base_font_size=10)


# (label, AUC, source-bucket)  — order = ranking (highest AUC first)
data = [
    ("ampel/snguess  (NEW v7)",    0.9497, "AMPEL"),
    ("fink/slsn  (NEW v7)",        0.9272, "Fink"),
    ("fink/snn",                   0.9207, "Fink"),
    ("pittgoogle/supernnova_lsst", 0.9121, "Pitt-Google"),
    ("supernnova",                 0.9067, "local rerun"),
    ("salt3_chi2",                 0.9026, "local rerun"),
    ("lc_features_bv",             0.8810, "local rerun"),
    ("alerce_lc",                  0.8790, "local rerun"),
    ("fink_lsst/early_snia",       0.8615, "Fink"),
    ("fink_lsst/cats",             0.8404, "Fink"),
    ("fink_lsst/snn",              0.8353, "Fink"),
    ("fink/rf_ia",                 0.8106, "Fink"),
]

palette = {
    "AMPEL": PALETTE["blue"],
    "Fink": PALETTE["black"],
    "Pitt-Google": PALETTE["green"],
    "local rerun": PALETTE["gray"],
}

labels  = [d[0] for d in data]
aucs    = [d[1] for d in data]
sources = [d[2] for d in data]
colors  = [palette[s] for s in sources]

# Reverse so highest AUC is at top of plot
labels = labels[::-1]
aucs = aucs[::-1]
colors = colors[::-1]
sources = sources[::-1]

fig, ax = plt.subplots(figsize=(6.9, 4.0))
y = list(range(len(labels)))
ax.barh(y, aucs, color=colors, height=0.62, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(0.78, 0.965)
ax.set_xlabel("Calibrated test ROC-AUC")
ax.set_title("Twelve calibrated trust heads ranked by held-out AUC", loc="left", pad=8)
ax.axvline(0.90, color=PALETTE["light_gray"], lw=0.9, zorder=1)

# Numeric labels on each bar
for yi, v in zip(y, aucs):
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=8.5, color=PALETTE["black"])

# Legend by broker source
import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color=c, label=k) for k, c in palette.items()]
ax.legend(handles=legend_handles, loc="lower right", ncol=2, columnspacing=1.0, handlelength=1.0)

clean_axis(ax, grid_axis="x")

plt.tight_layout()
out = Path(__file__).resolve().parent / "top12_auc.png"
save_figure(fig, out)
plt.close(fig)
print(f"Wrote {out} and {out.with_suffix('.pdf')}")
