"""Shared matplotlib style for ALL thesis figures (mémoire Polytechnique).

Usage in every plotting script:

    import thesis_style as ts
    ts.apply()
    ...
    ts.save(fig, outdir, "figure_name")   # writes .svg (thesis) + .png (preview)

Conventions
-----------
- Fonts: Latin Modern (same family as the LaTeX document), mathtext "cm".
- Palette: Okabe-Ito (colorblind-safe). Semantic roles are fixed across figures:
    BLUE    the "hero" series: filtered / safe / avec CBF / model under test
    ORANGE  nominal / baseline / sans CBF / danger, braking
    GREY    raw / reference / context (ground truth, obstacles, pre-filter)
    GREEN   auxiliary positive (poussee, recovery, ideal lines)
    PURPLE  modification / deviation magnitudes
- Legends NEVER inside the axes: always a horizontal strip above the axes,
  left-aligned (legend_top). Same place in every figure.
- Stats annotations via stats_box (uniform corner box).
- No in-figure titles: the LaTeX caption carries the description.
- Figures are sized at their FINAL printed width (\textwidth = 6.3 in) so the
  printed font size is uniform across the whole document.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
YELLOW = "#E69F00"
GREY = "#555555"
LIGHTGREY = "#909090"

# printed widths [inches]
TEXTWIDTH = 6.3


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",          # keep text as text in the SVG
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "axes.axisbelow": True,
    })


def save(fig, outdir, name):
    """Write <outdir>/<name>.svg (for the thesis) and .png (for quick preview)."""
    import os
    fig.savefig(os.path.join(outdir, name + ".svg"))
    fig.savefig(os.path.join(outdir, name + ".png"))
    plt.close(fig)


def panel_label(ax, s, x=-0.12):
    """Bold (a)/(b) corner label, consistent placement."""
    ax.text(x, 1.04, s, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="top")


def legend_top(ax, ncol=None, anchor=(0.0, 1.01), **kw):
    """The ONLY legend style: horizontal strip above the axes, left-aligned.
    Never overlaps data and sits in the same place in every figure."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ncol is None:
        ncol = min(len(labels), 4)
    return ax.legend(handles, labels, loc="lower left", bbox_to_anchor=anchor,
                     ncol=ncol, borderaxespad=0.0, columnspacing=1.2,
                     handlelength=1.5, handletextpad=0.5, **kw)


def fig_legend_top(fig, ax, ncol=4):
    """Figure-level variant for multi-panel figures sharing one legend."""
    handles, labels = ax.get_legend_handles_labels()
    return fig.legend(handles, labels, loc="lower left",
                      bbox_to_anchor=(0.02, 1.0), bbox_transform=fig.transFigure,
                      ncol=ncol, borderaxespad=0.0, columnspacing=1.2,
                      handlelength=1.5, handletextpad=0.5)


def stats_box(ax, text, loc="upper left"):
    """Uniform statistics box (same corner style in every figure)."""
    x, ha = (0.03, "left") if loc == "upper left" else (0.97, "right")
    ax.text(x, 0.95, text, transform=ax.transAxes, va="top", ha=ha, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#888888", lw=0.6, alpha=0.95))
