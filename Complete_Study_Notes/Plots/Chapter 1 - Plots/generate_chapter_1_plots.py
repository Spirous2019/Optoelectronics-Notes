"""
generate_chapter_1_plots.py
===========================
Generates all matplotlib figures for Chapter 1 of the Optoelectronics notes.

Currently produces:
  - 9.jpg  :  The Susceptibility Line Shapes (chi', chi'') vs normalised detuning δ,
               with the three key detuning cases annotated.

Run this script from any working directory.  The output image is saved
alongside this script in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Output directory is the same folder as this script ─────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette (figure-generation-style) ───────────────────────────────
WHITE    = "#ffffff"
AXES_CLR = "#2b2b2b"
GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"   # chi''  (absorption)
CORAL    = "#d94452"   # chi'   (dispersion)
GOLD     = "#c28800"   # delta = 0 annotation
LAVENDER = "#7e57c2"   # |delta| = 1 annotations
SKYBLUE  = "#1976d2"   # |delta| >> 1 band

# ── Global rcParams ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "axes.facecolor":      WHITE,
    "figure.facecolor":    WHITE,
    "axes.edgecolor":      AXES_CLR,
    "axes.labelcolor":     AXES_CLR,
    "xtick.color":         AXES_CLR,
    "ytick.color":         AXES_CLR,
    "text.color":          AXES_CLR,
    "axes.titlesize":      15,
    "axes.labelsize":      13,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     11,
    "lines.linewidth":     2.5,
    "grid.linestyle":      ":",
    "grid.linewidth":      0.6,
    "grid.alpha":          0.7,
    "grid.color":          GRID_CLR,
})

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 9  —  Susceptibility profiles χ'(δ) and χ''(δ)
# ═══════════════════════════════════════════════════════════════════════════

def plot_susceptibility_lineshapes():
    """
    χ'(δ)  = -δ / (1 + δ²)     [dispersive / real part]
    χ''(δ) = -1 / (1 + δ²)     [absorptive / imaginary part]

    Both are normalised so their extrema sit at ±½ and -1 respectively.
    The plot annotates the three detuning regimes discussed in the keyinsight:
      • δ = 0   : χ' = 0 (zero crossing), χ'' peaks at -1
      • |δ| = 1 : χ'' = -½ (half-max), χ' peaks at ±½
      • |δ| ≫ 1 : both → 0 (far off-resonance)
    """

    δ = np.linspace(-5, 5, 4000)

    chi_prime  = -δ       / (1 + δ**2)   # dispersive   (real part)
    # Plotting the magnitude of absorption so it peaks upward, as is standard
    chi_dprime = 1.0      / (1 + δ**2)   # absorptive   (imaginary part magnitude)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    # ── Curves ──────────────────────────────────────────────────────────────
    ax.plot(δ, chi_prime,  color=CORAL,  linestyle="-", linewidth=2.5,
            label=r"Dispersive Refraction ($\chi'$)")
    ax.plot(δ, chi_dprime, color=TEAL,   linestyle="--",  linewidth=2.5,
            label=r"Lorentzian Absorption ($|\chi''|$)")

    # ── Reference lines ─────────────────────────────────────────────────────
    ax.axhline(0,  color=AXES_CLR, linewidth=0.8, linestyle="-")
    ax.axvline(0,  color=GOLD,     linewidth=1.2, linestyle=":",  label=r"Resonance ($\delta=0$)")
    ax.axvline( 1, color=LAVENDER, linewidth=1.2, linestyle=":",  label=r"$|\delta|=1$ (half-absorption)")
    ax.axvline(-1, color=LAVENDER, linewidth=1.2, linestyle=":")

    # ── Regime shading: |δ| ≫ 1 ────────────────────────────────────────────
    ax.axvspan( 3.5,  5, alpha=0.08, color=SKYBLUE)
    ax.axvspan(-5, -3.5, alpha=0.08, color=SKYBLUE,
               label=r"$|\delta|\gg 1$ (off-resonance, both $\to 0$)")

    # ── Concise Annotations ─────────────────────────────────────────────────

    # Absorption peak
    ax.plot(0, 1.0, 'o', color=TEAL, markersize=7)
    ax.annotate(
        r"$|\chi''|_{\max}$",
        xy=(0, 1.0),
        xytext=(-0.5, 1.08),
        fontsize=12, color=TEAL,
        ha="center", va="center"
    )

    # Dispersion positive peak
    ax.plot(-1.0, 0.5, 'o', color=CORAL, markersize=7)
    ax.annotate(
        r"$\chi'_{\max}$",
        xy=(-1.0, 0.5),
        xytext=(-1.6, 0.6),
        fontsize=12, color=CORAL,
        ha="center", va="center"
    )

    # Dispersion negative peak
    ax.plot(1.0, -0.5, 'o', color=CORAL, markersize=7)
    ax.annotate(
        r"$-\chi'_{\max}$",
        xy=(1.0, -0.5),
        xytext=(1.6, -0.6),
        fontsize=12, color=CORAL,
        ha="center", va="center"
    )

    # ── FWHM double-headed arrow on χ'' ─────────────────────────────────────
    # |χ''| = ½ at δ = ±1  →  FWHM spans δ ∈ [-1, +1]
    fwhm_y = 0.5
    ax.annotate(
        "", xy=(1, fwhm_y), xytext=(-1, fwhm_y),
        arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.8),
    )
    ax.text(0, fwhm_y + 0.04, r"FWHM",
            fontsize=9, color=TEAL, ha="center", va="bottom")

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.85, 1.15)
    ax.set_xlabel(r"Normalised Detuning $\delta = 2(\omega - \omega_0)/\eta$", fontsize=13)
    ax.set_ylabel(r"Susceptibility Component (normalised)", fontsize=13)
    ax.set_title("The Fundamental Susceptibility Line Shapes", fontsize=15, pad=12)
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # Custom x-tick labels to highlight the key δ values
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    ax.set_xticklabels([r"$-5$", r"$-4$", r"$-3$", r"$-2$", r"$-1$",
                         r"$0$",  r"$1$",  r"$2$",  r"$3$",  r"$4$",  r"$5$"])

    legend = ax.legend(loc="upper right", framealpha=1,
                       facecolor="#f5f5f5", edgecolor="#cccccc",
                       labelcolor=AXES_CLR)

    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "9.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_susceptibility_lineshapes()
