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
# \textwidth du gabarit Polytechnique = 472.03 pt / 72.27 = 6.53 in.
# Pour que la police imprimee soit EXACTEMENT font.size (12 pt) quelle que
# soit la figure, on cree chaque figure a sa largeur physique finale (en
# pouces) et on l'inclut SANS option width :  \includesvg[inkscapelatex=false]{...}
# Ainsi l'echelle vaut toujours 1 et 12 pt matplotlib = 12 pt sur la page.
# La fraction de page occupee se regle ici, via figsize, pas dans LaTeX.
TEXTWIDTH = 6.53        # pleine largeur de texte
def width(frac=1.0):
    """Largeur physique [pouces] pour une figure occupant `frac` de \\textwidth.
    A inclure ensuite sans `width` dans LaTeX pour garder l'echelle = 1."""
    return frac * TEXTWIDTH


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",          # keep text as text in the SVG
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Police de base = 12 pt, identique au corps de texte du mémoire.
        # Valable seulement si la figure est créée à TEXTWIDTH et incluse
        # à width=\textwidth (échelle 1). Sinon, multiplier par l'échelle.
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
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
    Never overlaps data and sits in the same place in every figure.

    The strip wraps onto extra rows when it would be wider than the axes:
    ``ncol`` is treated as an upper bound and reduced until the legend fits,
    so long labels never spill past the right edge of the graph."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    max_ncol = len(labels) if ncol is None else min(ncol, len(labels))

    def build(nc):
        return ax.legend(handles, labels, loc="lower left", bbox_to_anchor=anchor,
                         ncol=nc, borderaxespad=0.0, columnspacing=1.2,
                         handlelength=1.5, handletextpad=0.5, **kw)

    fig = ax.figure
    try:
        fig.canvas.draw()  # need a renderer to measure widths
        ax_w = ax.get_window_extent().width
        for nc in range(max_ncol, 0, -1):
            leg = build(nc)
            fig.canvas.draw()
            if leg.get_window_extent().width <= ax_w or nc == 1:
                return leg
            leg.remove()
    except Exception:
        pass
    return build(max_ncol)


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
