"""Shared NeurIPS-style plotting helpers for metaDEBASS figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# Okabe-Ito / colorblind-safe palette with a small grayscale hierarchy.
PALETTE = {
    "black": "#111827",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
}


def apply_neurips_style(*, base_font_size: int = 9, serif: bool = False) -> None:
    """Apply a restrained conference-figure Matplotlib style."""
    family = "serif" if serif else "sans-serif"
    plt.rcParams.update({
        "font.family": family,
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "axes.linewidth": 0.75,
        "axes.edgecolor": PALETTE["black"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "grid.color": PALETTE["light_gray"],
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def save_figure(fig, png_path: Path, *, pdf_path: Path | None = None, dpi: int = 300) -> None:
    """Save a high-DPI PNG and a vector PDF sibling by default."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)


def clean_axis(ax, *, grid_axis: str | None = "y") -> None:
    """Apply common axis cleanup after plotting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
