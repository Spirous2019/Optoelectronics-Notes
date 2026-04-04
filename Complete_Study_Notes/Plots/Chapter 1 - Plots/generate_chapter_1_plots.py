"""
generate_chapter_1_plots.py
===========================
Generates all matplotlib figures for Chapter 1 of the Optoelectronics notes.

Currently produces:
  - 9.jpg  :  The Susceptibility Line Shapes (chi', chi'') vs normalised detuning δ,
               with the three key detuning cases annotated.
  - 14.jpg :  The Output Transmission Spectrum with two Lorentzian absorption dips,
               FWHM bandwidth shaded regions replacing the old arrow annotations.

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


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 14  —  Output Transmission Spectrum with shaded absorption bands
# ═══════════════════════════════════════════════════════════════════════════

def plot_transmission_spectrum():
    """
    T(ν) = exp(−α₁(ν) − α₂(ν))  where αᵢ(ν) is a Lorentzian.

    Output is saved to the same folder as this script for manual revision
    before the user moves it to Figures/Chapter 1/.

    Clean presentation: shaded FWHM bands communicate that absorption lines
    are finite-bandwidth regions. No arrows, no callout labels, no centre lines.
    """
    nu = np.linspace(0.0, 11.0, 6000)

    # ── Resonance parameters ─────────────────────────────────────────────────
    nu0_1, dnu_1, aL_1 = 3.0, 0.50, 1.50   # ν₀, FWHM, α_peak·L
    nu0_2, dnu_2, aL_2 = 7.5, 0.70, 2.50

    # Lorentzian absorption profiles and total transmission
    alpha_1 = aL_1 / (1.0 + ((nu - nu0_1) / (dnu_1 / 2.0))**2)
    alpha_2 = aL_2 / (1.0 + ((nu - nu0_2) / (dnu_2 / 2.0))**2)
    T = np.exp(-(alpha_1 + alpha_2))

    # ── FWHM band boundaries (ν₀ ± Δν/2) ───────────────────────────────────
    b1_lo, b1_hi = nu0_1 - dnu_1 / 2, nu0_1 + dnu_1 / 2   # 2.75 – 3.25
    b2_lo, b2_hi = nu0_2 - dnu_2 / 2, nu0_2 + dnu_2 / 2   # 7.15 – 7.85

    # ── Figure setup ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    # ── Transmission curve ───────────────────────────────────────────────────
    ax.plot(nu, T, color=TEAL, linewidth=2.5, label="Transmission Spectrum")

    # ── FWHM shading (drawn first so the curve sits on top) ─────────────────
    ax.axvspan(b1_lo, b1_hi, alpha=0.11, color=CORAL,
               label=r"Absorption Bandwidth ($\Delta\nu$ FWHM region)")
    ax.axvspan(b2_lo, b2_hi, alpha=0.11, color=CORAL)

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks([nu0_1, nu0_2])
    ax.set_xticklabels([r"$\nu_{0,1}$", r"$\nu_{0,2}$"], fontsize=12)
    ax.set_xlabel(r"Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Output Transmission $T(\nu) = e^{-\alpha(\nu)L}$", fontsize=13)
    ax.set_title("The Output Transmission Spectrum", fontsize=15, pad=12)
    ax.grid(True)

    legend = ax.legend(loc="upper right", framealpha=1,
                       facecolor="#f5f5f5", edgecolor="#cccccc",
                       labelcolor=AXES_CLR)

    fig.tight_layout()

    # Save to same folder as script for revision before manual move
    out_path = os.path.join(OUT_DIR, "14.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_susceptibility_lineshapes()
    plot_transmission_spectrum()
