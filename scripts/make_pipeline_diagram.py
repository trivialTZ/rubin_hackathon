"""
DEBASS end-to-end pipeline diagram — publication style.
Run: python scripts/make_pipeline_diagram.py
Output: docs/slides_figures/fig_pipeline.{png,pdf}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs" / "slides_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":    8,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

# ── Canvas ────────────────────────────────────────────────────────────
XW, YH = 10.0, 26.0
fig, ax = plt.subplots(figsize=(8.5, 22.1))   # ≈ 8.5 × (22.1/8.5×10) units
ax.set_xlim(0, XW)
ax.set_ylim(0, YH)
ax.axis("off")

# ── Section palette ───────────────────────────────────────────────────
# (background, border/title)
PAL = {
    "col": ("#D6EAF8", "#2874A6"),   # blue
    "tru": ("#D5F5E3", "#1E8449"),   # green
    "fea": ("#FEFDE7", "#9A7D0A"),   # amber
    "tra": ("#F4ECF7", "#7D3C98"),   # purple
    "sco": ("#FDEDEC", "#CB4335"),   # red
}

# ── Helpers ───────────────────────────────────────────────────────────

def panel(y0, y1, key, title):
    bg, bd = PAL[key]
    ax.add_patch(FancyBboxPatch(
        (0.18, y0), XW - 0.36, y1 - y0,
        boxstyle="round,pad=0.08", fc=bg, ec=bd, lw=1.3, zorder=1))
    ax.text(XW / 2, y1 - 0.22, title,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=bd, zorder=5)


def node(cx, cy, w, h, txt,
         fs=7.5, fc="white", ec="0.3", tc="black",
         bold=False, mono=False, lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.07", fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, txt,
            ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            fontfamily="monospace" if mono else "serif",
            zorder=4, multialignment="center", linespacing=1.35)


def arr(x0, y0, x1, y1, label="", lc="0.25", lside="right", lfs=6.5):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=lc, lw=0.8,
                        connectionstyle="arc3,rad=0"),
        zorder=6)
    if label:
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        dx = 0.12 if lside == "right" else -0.12
        ax.text(xm + dx, ym, label,
                ha="left" if lside == "right" else "right",
                va="center", fontsize=lfs, color=lc, style="italic", zorder=7)


# ═════════════════════════════════════════════════════════════════════
# S1  DATA COLLECTION   y = 20.5 → 26.0
# ═════════════════════════════════════════════════════════════════════
panel(20.5, 26.0, "col", "DATA COLLECTION")

# labels.csv
node(2.2, 25.35, 3.0, 0.55,
     "labels.csv\n(object_id, ra, dec, broker, …)",
     fs=7, fc="#D6EAF8", ec="#2874A6", mono=True)
arr(3.7, 25.35, 9.5, 25.35,
    label="drives  backfill.py  --parallel 8", lc="#2874A6")

# Broker boxes
BROKERS = [
    (2.0,  "ALeRCE\n(ZTF + LSST)\nv2.1 API  ·  --parallel 8"),
    (5.2,  "Fink\n(ZTF + LSST)\nper-alert scores\nSNN / CATS / EarlySNIa"),
    (8.3,  "Lasair\n(ZTF + LSST)\nSherlock\nhost context"),
]
BKW, BKH, BKY = 3.0, 1.65, 23.6
for bx, txt in BROKERS:
    node(bx, BKY, BKW, BKH, txt, fs=7.2, fc="white", ec="#2874A6")
    arr(bx, BKY - BKH / 2, bx, 22.3, lc="0.3")

# Bronze
BRONZY = 22.05
node(5.0, BRONZY, 9.0, 0.6,
     "Bronze  ·  raw broker payloads  ·  data/bronze/*.parquet",
     fs=7.2, fc="#FDFEFF", ec="#5D6D7E", mono=True)
arr(5.0, BRONZY - 0.30, 5.0, 21.4,
    label="normalize.py", lc="0.3")

# Silver
SILVY = 21.15
node(5.0, SILVY, 9.0, 0.75,
     "Silver  ·  event-level, exactness-tagged\n"
     "data/silver/broker_events.parquet   +   data/silver/local_expert_outputs/",
     fs=7, fc="#FDFEFF", ec="#5D6D7E", mono=True)

arr(5.0, SILVY - 0.38, 5.0, 20.52, lc="0.3")  # exit section 1

# ═════════════════════════════════════════════════════════════════════
# S2  TRUTH LABELS   y = 14.2 → 20.3
# ═════════════════════════════════════════════════════════════════════
panel(14.2, 20.3, "tru", "TRUTH LABELS")

# Three source boxes on the left
SRCS = [
    (19.4, "TNS bulk CSV\n(160 K objects)"),
    (18.1, "Fink LSST  xm_*\n(free per-alert:\n tns_type · simbad_otype\n photo-z · Gaia Plx)"),
    (16.6, "Host context\n(SIMBAD = galaxy →\n likely SN host;\n NOT contamination)"),
]
for sy, stxt in SRCS:
    node(1.9, sy, 3.0, 1.0, stxt, fs=7, fc="white", ec="#1E8449")

# crossmatch_tns.py (inline step from TNS)
node(5.5, 19.4, 2.8, 0.75,
     "crossmatch_tns.py\n(name + positional fallback)", fs=7, fc="white", ec="#1E8449")
arr(3.4, 19.4, 4.1, 19.4, lc="0.3")                  # TNS → crossmatch

# build_truth_multisource.py  (large box, right side)
BTY, BTH = 17.5, 4.6
node(7.8, BTY, 3.8, BTH,
     "build_truth_multisource.py\n\n"
     "→ data/truth/object_truth.parquet\n\n"
     "5-tier truth quality:\n"
     "  1. spectroscopic  (TNS type)\n"
     "  2. tns_untyped  (AT prefix)\n"
     "  3. broker consensus  (≥ 2 agree)\n"
     "  4. host context  (SIMBAD + photo-z)\n"
     "  5. weak  (broker self-label)",
     fs=7, fc="white", ec="#1E8449")

BT_LEFT = 7.8 - 3.8 / 2   # = 5.9
arr(6.9, 19.4, BT_LEFT, 19.0, lc="0.3")              # crossmatch → build
arr(3.4, 18.1, BT_LEFT, 18.0, lc="0.3")              # fink xm →
arr(3.4, 16.6, BT_LEFT, 16.8, lc="0.3")              # host context →

arr(5.0, 14.2, 5.0, 13.82, lc="0.3")                 # exit section 2

# ═════════════════════════════════════════════════════════════════════
# S3  FEATURE EXTRACTION   y = 9.0 → 13.7
# ═════════════════════════════════════════════════════════════════════
panel(9.0, 13.7, "fea", "FEATURE EXTRACTION")

# Horizontal chain: lightcurves → detection.py → lightcurve.py
CHY = 13.0
CHAIN = [
    (1.4,  "Lightcurves\n(*.json)"),
    (3.8,  "detection.py\n(ZTF: fid / mag\nLSST: flux / band\n→ common schema)"),
    (7.0,  "lightcurve.py\n(no-leakage: truncate\nto exactly n_det;\n51 features)"),
]
for cx, txt in CHAIN:
    node(cx, CHY, 2.2, 1.4, txt, fs=7, fc="white", ec="#9A7D0A")
arr(2.5, CHY, 2.7, CHY, lc="0.3")
arr(4.9, CHY, 5.9, CHY, lc="0.3")

# Gold snapshot box
GY, GH = 11.0, 2.5
node(5.0, GY, 9.0, GH,
     "Gold  ·  object-epoch snapshots\n\n"
     "For each (object_id,  n_det = 1 … 20):\n"
     "  •  51 LC features           (no-leakage truncated lightcurve)\n"
     "  •  survey flag              (survey_is_lsst:  ZTF = 0,  LSST = 1)\n"
     "  •  18 expert projections    (p_snia, p_nonIa, p_other per expert)\n"
     "  •  truth label              (if available)\n\n"
     "data/gold/object_epoch_snapshots.parquet",
     fs=7, fc="#FEFBE8", ec="#9A7D0A", mono=False)

arr(7.0, CHY - 0.7, 6.5, GY + GH / 2, lc="0.3")     # chain → gold

arr(5.0, GY - GH / 2, 5.0, 9.02, lc="0.3")           # exit section 3

# ═════════════════════════════════════════════════════════════════════
# S4  MODEL TRAINING   y = 2.2 → 8.8
# ═════════════════════════════════════════════════════════════════════
panel(2.2, 8.8, "tra", "MODEL TRAINING")

# Expert helpfulness
HY = 7.9
node(5.0, HY, 6.5, 1.0,
     "Expert Helpfulness Table\n"
     "is_topclass_correct at each epoch?   ·   data/gold/expert_helpfulness.parquet",
     fs=7, fc="white", ec="#7D3C98")
arr(5.0, HY - 0.5, 5.0, 6.95, lc="0.4")

# Trust heads
TY, TH = 5.9, 2.3
node(5.0, TY, 6.5, TH,
     "Per-Expert Trust Heads  (×6 trained  ·  ×18 registered)\n\n"
     "LightGBM binary classifier — one per expert\n"
     "Input :  51 LC features  +  expert projection  +  availability flags\n"
     "Target:  is_topclass_correct\n"
     "Output:  trust  q  in  [0, 1]\n\n"
     "→  models/trust/{expert_key}/model.pkl",
     fs=7, fc="white", ec="#7D3C98")
arr(5.0, TY - TH / 2, 5.0, 4.4, lc="0.4")

# Follow-up head
FUY, FUH = 3.5, 2.1
node(5.0, FUY, 6.5, FUH,
     "Follow-Up Head  (shared model)\n\n"
     "Input :  all trust scores  q_i  +  expert projections  +  51 LC features\n"
     "Target:  target_follow_proxy\n"
     "Output:  p_follow  in  [0, 1]          ROC-AUC = 0.9949  (held-out test)\n\n"
     "→  models/followup/model.pkl",
     fs=7, fc="white", ec="#7D3C98")

arr(5.0, FUY - FUH / 2, 5.0, 2.22, lc="0.4")         # exit section 4

# ═════════════════════════════════════════════════════════════════════
# S5  SCORING OUTPUT   y = 0.2 → 2.0
# ═════════════════════════════════════════════════════════════════════
panel(0.2, 2.0, "sco", "SCORING OUTPUT")

node(5.0, 1.05, 9.0, 1.3,
     "score_nightly.py\n\n"
     "Per expert:  trust  ·  projected probs  ·  mapped_pred_class\n"
     "Ensemble:  p_snia_weighted (trust-weighted avg)  ·  "
     "p_follow_proxy (learned)  ·  recommended (yes / no)\n"
     "→  reports/scores/*.jsonl",
     fs=7, fc="white", ec="#CB4335")

# ─────────────────────────────────────────────────────────────────────
out_png = OUT / "fig_pipeline.png"
out_pdf = OUT / "fig_pipeline.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
