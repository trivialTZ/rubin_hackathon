"""Generate a deterministic NeurIPS-style metaDEBASS architecture figure."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.features.lightcurve import FEATURE_NAMES
from debass_meta.projectors.base import ALL_EXPERT_KEYS
from metadebass_plot_style import PALETTE, apply_neurips_style, save_figure

OUT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs" / "slides_figures"
IMAGEGEN_SOURCE = OUT_DIR / "architecture_v7_imagegen.png"

N_FEATURES = len(FEATURE_NAMES)
N_REGISTERED = len(ALL_EXPERT_KEYS)
N_CALIBRATED_V7 = 12

assert N_FEATURES == 51, f"Expected 51 LC features, got {N_FEATURES}"
assert N_REGISTERED == 28, f"Expected 28 registered experts, got {N_REGISTERED}"
assert N_CALIBRATED_V7 == 12

BLACK = PALETTE["black"]
GRAY = PALETTE["gray"]
LIGHT = "#F8FAFC"
RULE = "#CBD5E1"


def _save_png_as_pdf(png_path: Path, pdf_path: Path) -> None:
    with Image.open(png_path) as im:
        im.convert("RGB").save(pdf_path, "PDF", resolution=300.0)


def _install_imagegen_asset() -> bool:
    """Install the audited imagegen architecture asset when present."""
    if not IMAGEGEN_SOURCE.exists():
        return False

    targets = [
        (OUT_DIR / "architecture_v7.png", OUT_DIR / "architecture_v7.pdf"),
        (DOCS_DIR / "fig5_architecture.png", DOCS_DIR / "fig5_architecture.pdf"),
    ]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for png_path, pdf_path in targets:
        shutil.copyfile(IMAGEGEN_SOURCE, png_path)
        _save_png_as_pdf(png_path, pdf_path)
        print(f"Wrote {png_path}")
    return True


def label(ax, x, y, text, *, size=8.5, weight="normal", color=BLACK, ha="left", va="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.25,
        zorder=8,
    )


def rounded_panel(ax, x, y, w, h, *, fc="white", ec=RULE, lw=0.8, radius=0.08, zorder=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        fc=fc,
        ec=ec,
        lw=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def module(ax, x, y, w, h, title, body, *, accent, fill=LIGHT, body_size=8.1):
    """Draw a compact paper-figure module with a thin accent rule."""
    rounded_panel(ax, x, y, w, h, fc=fill, ec=RULE, lw=0.75, radius=0.055, zorder=3)
    ax.add_patch(Rectangle((x, y + h - 0.065), w, 0.065, fc=accent, ec="none", zorder=4))
    label(ax, x + 0.15, y + h - 0.24, title, size=8.8, weight="bold", color=BLACK, va="top")
    label(ax, x + 0.15, y + 0.22, body, size=body_size, color=BLACK, va="bottom")


def arrow(ax, start, end, *, color=GRAY, lw=0.9, mutation=10, zorder=5, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        lw=lw,
        color=color,
        shrinkA=2.5,
        shrinkB=2.5,
        connectionstyle="arc3,rad=0",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def pill(ax, x, y, w, h, text, *, fc, ec=None, text_color="white", size=7.5):
    rounded_panel(ax, x, y, w, h, fc=fc, ec=ec or fc, lw=0.6, radius=h / 2, zorder=7)
    label(ax, x + w / 2, y + h / 2, text, size=size, weight="bold", color=text_color, ha="center")


def main() -> None:
    if _install_imagegen_asset():
        return

    apply_neurips_style(base_font_size=9)
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Compact standalone title so the figure also works outside the deck.
    label(ax, 0.45, 4.9, "metaDEBASS trust-aware early-epoch architecture", size=13.0, weight="bold", va="top")
    label(
        ax,
        0.45,
        4.55,
        "Primary product: expert_confidence at (object_id, n_det=N, alert_jd); follow-up proxy is downstream.",
        size=8.7,
        color=GRAY,
        va="top",
    )

    # Temporal safety lane: shown as a rule in the computation, not as a bulky callout.
    guard_x, guard_y, guard_w, guard_h = 2.35, 3.72, 8.20, 0.54
    rounded_panel(ax, guard_x, guard_y, guard_w, guard_h, fc="#FFFBF0", ec="#F1B84B", lw=0.8, radius=0.07, zorder=1)
    ax.add_patch(Rectangle((guard_x, guard_y), 0.08, guard_h, fc=PALETTE["orange"], ec="none", zorder=2))
    label(ax, guard_x + 0.23, guard_y + guard_h / 2, "Temporal safety gate:", size=8.8, weight="bold", color="#8A5A00")
    label(
        ax,
        guard_x + 1.92,
        guard_y + guard_h / 2,
        "preserve event_time_jd, alert_id, event_scope, temporal_exactness;  latest_object_unsafe excluded",
        size=8.1,
        color=BLACK,
    )

    y, h = 2.05, 1.24
    stages = [
        (0.55, 1.45, "Object epoch", "object_id\nn_det = N\nalert_jd", PALETTE["blue"], 8.4),
        (2.28, 1.55, "LC features", f"detections 1..N\n{N_FEATURES} features\nno future rows", PALETTE["blue"], 8.2),
        (
            4.10,
            1.80,
            "Event-safe as-of join",
            "event_time_jd\ntemporal_exactness\nlatest_object_unsafe\nexcluded",
            PALETTE["orange"],
            7.1,
        ),
        (
            6.20,
            1.70,
            "Expert projections",
            f"{N_REGISTERED} registered experts\nternary evidence\nor context",
            PALETTE["purple"],
            7.95,
        ),
        (
            8.20,
            1.82,
            "Trust heads",
            f"{N_CALIBRATED_V7} calibrated v7\nLightGBM + isotonic\nnative NaN",
            BLACK,
            7.95,
        ),
    ]

    for x, w, title, body, accent, body_size in stages:
        module(ax, x, y, w, h, title, body, accent=accent, body_size=body_size)

    for (x0, w0, *_), (x1, *_rest) in zip(stages[:-1], stages[1:]):
        arrow(ax, (x0 + w0 + 0.05, y + h / 2), (x1 - 0.05, y + h / 2), lw=1.0)

    # Evidence input into the as-of join.
    ev_x, ev_y, ev_w, ev_h = 4.25, 0.58, 2.28, 0.96
    rounded_panel(ax, ev_x, ev_y, ev_w, ev_h, fc="white", ec=RULE, lw=0.75, radius=0.06, zorder=2)
    label(ax, ev_x + 0.14, ev_y + ev_h - 0.18, "Broker/local evidence", size=8.2, weight="bold", va="top")
    label(
        ax,
        ev_x + 0.14,
        ev_y + ev_h - 0.47,
        "ALeRCE, Fink, Lasair, Pitt-Google,\nANTARES, AMPEL, Babamul, reruns",
        size=7.15,
        color=BLACK,
        va="top",
    )
    arrow(ax, (ev_x + ev_w / 2, ev_y + ev_h + 0.04), (4.10 + 1.80 / 2, y - 0.04), color=GRAY, lw=0.85)
    label(ax, 5.08, 1.74, "event rows", size=7.2, color=GRAY, ha="center")

    # Safety lane alignment cues.
    arrow(ax, (5.00, guard_y), (5.00, y + h + 0.08), color=PALETTE["orange"], lw=0.75, mutation=8)
    arrow(ax, (9.02, guard_y), (9.02, y + h + 0.08), color=PALETTE["orange"], lw=0.75, mutation=8)

    # Output heads with strong primary hierarchy.
    out_x = 10.78
    primary_y, primary_w, primary_h = 2.62, 1.72, 0.76
    secondary_y, secondary_w, secondary_h = 1.46, 1.72, 0.64
    rounded_panel(ax, out_x, primary_y, primary_w, primary_h, fc="#ECFDF5", ec=PALETTE["green"], lw=1.0, radius=0.07, zorder=5)
    ax.add_patch(Rectangle((out_x, primary_y), 0.09, primary_h, fc=PALETTE["green"], ec="none", zorder=6))
    label(ax, out_x + 0.23, primary_y + primary_h / 2, "expert_confidence", size=9.2, weight="bold", color=BLACK)
    pill(ax, out_x + 0.96, primary_y + primary_h - 0.28, 0.55, 0.20, "PRIMARY", fc=PALETTE["green"], size=6.2)

    rounded_panel(ax, out_x + 0.12, secondary_y, secondary_w - 0.24, secondary_h, fc="#EFF6FF", ec=PALETTE["blue"], lw=0.85, radius=0.07, zorder=5)
    ax.add_patch(Rectangle((out_x + 0.12, secondary_y), 0.07, secondary_h, fc=PALETTE["blue"], ec="none", zorder=6))
    label(ax, out_x + 0.35, secondary_y + secondary_h / 2, "p_follow_proxy", size=8.8, weight="bold", color=BLACK)
    pill(ax, out_x + 1.08, secondary_y - 0.27, 0.66, 0.20, "secondary", fc=PALETTE["blue"], size=6.3)

    trust_x, trust_w = stages[-1][0], stages[-1][1]
    arrow(ax, (trust_x + trust_w + 0.08, y + h * 0.65), (out_x - 0.04, primary_y + primary_h / 2), color=BLACK, lw=1.05, mutation=11)
    arrow(ax, (trust_x + trust_w + 0.08, y + h * 0.38), (out_x + 0.08, secondary_y + secondary_h / 2), color=GRAY, lw=0.8, mutation=9)

    # Training contract note kept visually quiet.
    ax.plot([0.45, 12.45], [0.34, 0.34], color=RULE, lw=0.65, zorder=1)
    label(
        ax,
        0.45,
        0.15,
        "Object-group train/cal/test split; weak broker-derived labels remain marked weak; EarlyMeta is a benchmark only, not part of the trust stack.",
        size=7.8,
        color=GRAY,
    )

    out_png = OUT_DIR / "architecture_v7.png"
    save_figure(fig, out_png, pdf_path=OUT_DIR / "architecture_v7.pdf")
    save_figure(fig, DOCS_DIR / "fig5_architecture.png", pdf_path=DOCS_DIR / "fig5_architecture.pdf")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {DOCS_DIR / 'fig5_architecture.png'}")


if __name__ == "__main__":
    main()
