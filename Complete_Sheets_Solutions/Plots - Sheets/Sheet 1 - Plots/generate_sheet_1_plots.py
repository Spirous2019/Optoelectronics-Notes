"""
generate_sheet_1_plots.py
=========================
Generates all matplotlib figures for Sheet 1 of the Optoelectronics course.

Produces:
  - sheet1_susceptibility.jpg          : chi'(delta) and chi''(delta) lineshapes
                                         with extreme points and FWHM annotated.
  - sheet1_refractive_index_two_res.jpg: n(nu) for a medium with two resonant
                                         absorptions (UV and IR), as in Q4.

Run from any working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Paths ────────────────────────────────────────────────────────────────────
# Plots are saved next to this script for manual review.
# Once approved, copy them to Complete_Sheets_Solutions/Figures/Sheet 1/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette (unified figure-generation-style) ─────────────────────────
WHITE    = "#ffffff"
AXES_CLR = "#2b2b2b"
GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"
CORAL    = "#d94452"
GOLD     = "#c28800"
LAVENDER = "#7e57c2"
SKYBLUE  = "#1976d2"

# ── Global rcParams ──────────────────────────────────────────────────────────
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
#  FIGURE 1 — Susceptibility Line Shapes χ'(δ) and χ''(δ)   [Q1(D)]
# ═══════════════════════════════════════════════════════════════════════════

def plot_susceptibility():
    """
    chi'(delta)  = -delta / (1 + delta^2)     [dispersive / real part]
    chi''(delta) =   -1   / (1 + delta^2)     [absorptive / imaginary part]

    Extreme points:
      chi' :  max = +1/2 at delta = -1,  min = -1/2 at delta = +1
      |chi''|: max =  1   at delta =  0
    """
    delta = np.linspace(-5, 5, 5000)

    chi_p  = -delta / (1 + delta**2)          # chi'
    chi_pp = -1.0   / (1 + delta**2)          # chi''
    abs_chi_pp = np.abs(chi_pp)               # |chi''|

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    fig.patch.set_facecolor(WHITE)

    # ── Left panel: chi'(delta) ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(WHITE)
    ax.plot(delta, chi_p, color=CORAL, linewidth=2.5, label=r"$\chi'(\nu)$")
    ax.axhline(0, color=AXES_CLR, linewidth=0.8)
    ax.axvline(0, color=GOLD, linewidth=1.2, linestyle=":", label=r"$\nu = \nu_o$  ($\delta=0$)")

    # Extreme points: delta = -1 (max) and +1 (min)
    ax.axvline(-1, color=LAVENDER, linewidth=1.2, linestyle=":")
    ax.axvline( 1, color=LAVENDER, linewidth=1.2, linestyle=":",
                label=r"$|\delta|=1$  ($\nu_o \pm \Delta\nu/2$)")

    chi_p_max = -(-1) / (1 + 1)   # = +0.5
    chi_p_min = -(+1) / (1 + 1)   # = -0.5

    ax.plot(-1, chi_p_max, 'o', color=CORAL, markersize=8, zorder=6)
    ax.annotate(r"$\chi'_{\max} = \dfrac{\chi_o}{2\nu_o\Delta\nu}$",
                xy=(-1, chi_p_max), xytext=(-3.5, chi_p_max - 0.1),
                fontsize=10, color=CORAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0))

    ax.plot(1, chi_p_min, 'o', color=CORAL, markersize=8, zorder=6)
    ax.annotate(r"$\chi'_{\min} = -\dfrac{\chi_o}{2\nu_o\Delta\nu}$",
                xy=(1, chi_p_min), xytext=(2.5, chi_p_min + 0.1),
                fontsize=10, color=CORAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([r"$\nu_o - \frac{\Delta\nu}{2}$",
                        r"$\nu_o$",
                        r"$\nu_o + \frac{\Delta\nu}{2}$"], fontsize=10)
    ax.set_xlabel(r"Normalised Detuning $\delta = 4\pi(\nu-\nu_o)/\sigma$", fontsize=12)
    ax.set_ylabel(r"$\chi'(\nu)$  (normalised)", fontsize=12)
    ax.set_title(r"Dispersive Part $\chi'(\nu)$", fontsize=14, pad=10)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", fontsize=10)

    # ── Right panel: |chi''(delta)| ──────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(WHITE)
    ax.plot(delta, abs_chi_pp, color=TEAL, linewidth=2.5, label=r"$|\chi''(\nu)|$")
    ax.axhline(0, color=AXES_CLR, linewidth=0.8)
    ax.axvline(0, color=GOLD, linewidth=1.2, linestyle=":", label=r"$\nu = \nu_o$  (peak)")

    # Half-power points
    fwhm_y = 0.5
    ax.axvline(-1, color=LAVENDER, linewidth=1.2, linestyle=":")
    ax.axvline( 1, color=LAVENDER, linewidth=1.2, linestyle=":",
                label=r"FWHM boundaries  $\Delta\nu = \sigma/2\pi$")
    ax.annotate("", xy=(1, fwhm_y), xytext=(-1, fwhm_y),
                arrowprops=dict(arrowstyle="<->", color=LAVENDER, lw=1.8))
    ax.text(0, fwhm_y + 0.04, r"$\Delta\nu$", fontsize=11,
            color=LAVENDER, ha="center", va="bottom")

    # Peak dot
    ax.plot(0, 1.0, 'o', color=TEAL, markersize=8, zorder=6)
    ax.annotate(r"$|\chi''|_{\max} = \dfrac{\chi_o}{\nu_o\Delta\nu}$",
                xy=(0, 1.0), xytext=(1.8, 0.88),
                fontsize=10, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([r"$\nu_o - \frac{\Delta\nu}{2}$",
                        r"$\nu_o$",
                        r"$\nu_o + \frac{\Delta\nu}{2}$"], fontsize=10)
    ax.set_xlabel(r"Normalised Detuning $\delta = 4\pi(\nu-\nu_o)/\sigma$", fontsize=12)
    ax.set_ylabel(r"$|\chi''(\nu)|$  (normalised)", fontsize=12)
    ax.set_title(r"Absorptive Part $|\chi''(\nu)|$", fontsize=14, pad=10)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", fontsize=10)

    fig.suptitle(r"Susceptibility Line Shapes of the Classical Electron Oscillator",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "q1d_susceptibility.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Refractive Index n(nu) for Two Resonances   [Q4]
# ═══════════════════════════════════════════════════════════════════════════

def plot_two_resonance_index():
    """
    n(nu) = n_o + chi'(nu) / (2*n_o)
    where chi' is the sum of two dispersive contributions centred
    at nu_01 (UV/blue) and nu_02 (mid-IR).

    The x-axis runs in wavelength (lambda), so higher nu = lower lambda:
    the UV resonance appears on the LEFT, the IR on the RIGHT.
    """
    # Resonance parameters (arbitrary normalised units in frequency)
    nu_01 = 8.0    # UV resonance  (high frequency, short wavelength)
    nu_02 = 2.5    # IR resonance  (low frequency,  long  wavelength)
    dnu   = 0.25   # linewidth (same for both, narrow)
    chi_o = 0.6    # peak susceptibility amplitude (same for both)
    n_o   = 1.5    # background refractive index

    nu = np.linspace(0.5, 11.5, 8000)

    def chi_prime(f, f0):
        """Near-resonance dispersive lineshape."""
        delta = 4 * np.pi * (f - f0) / (dnu * 2 * np.pi)   # keep consistent with text
        return -(chi_o / (n_o * dnu)) * delta / (1 + delta**2)

    n = n_o + chi_prime(nu, nu_01) + chi_prime(nu, nu_02)

    # Clip wild excursions right at the singularity for cleaner plotting
    n_clipped = np.clip(n, n_o - 1.8, n_o + 1.8)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    ax.plot(nu, n_clipped, color=TEAL, linewidth=2.5, label=r"$n(\nu)$")

    # Resonance vertical dashed lines
    ax.axvline(nu_01, color=CORAL, linewidth=1.4, linestyle="--",
               label=r"Resonances $\nu_{0_1}$ (UV),  $\nu_{0_2}$ (IR)")
    ax.axvline(nu_02, color=CORAL, linewidth=1.4, linestyle="--")

    # Asymptotic level lines
    n_inf   = n_o                                       # nu -> infinity
    n_mid   = n_o + chi_o / (n_o * dnu) * 0.0          # at resonance, chi'=0
    # far below nu_02 (nu->0): n asymptotes to n_o + both contributions at delta -> -inf
    # effectively each chi' -> 0 for delta -> ±inf, so n -> n_o everywhere far from res.
    ax.axhline(n_o, color=AXES_CLR, linewidth=1.0, linestyle=":",
               label=r"$n_o$ (background, far from resonances)")

    # Annotations
    ax.text(nu_01, ax.get_ylim()[0] + 0.15 if ax.get_ylim()[0] > 0 else 0.15,
            r"$\nu_{0_1}$ (UV)", fontsize=11, color=CORAL, ha="center", va="bottom")
    ax.text(nu_02, 0.15,
            r"$\nu_{0_2}$ (IR)", fontsize=11, color=CORAL, ha="center", va="bottom")

    # Normal vs anomalous dispersion annotation
    ax.annotate("Normal dispersion\n" + r"$(\partial n/\partial\nu > 0)$",
                xy=(5.0, n_o + 0.05), fontsize=10, color=SKYBLUE, ha="center")
    ax.annotate("Anomalous\ndispersion",
                xy=(nu_01 + 0.6, n_o + 0.55), fontsize=9, color=GOLD, ha="left")
    ax.annotate("Anomalous\ndispersion",
                xy=(nu_02 + 0.3, n_o - 0.55), fontsize=9, color=GOLD, ha="left")

    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(n_o - 1.5, n_o + 1.5)
    ax.set_xticks([nu_02, nu_01])
    ax.set_xticklabels([r"$\nu_{0_2}$ (IR)", r"$\nu_{0_1}$ (UV)"], fontsize=12)
    ax.set_xlabel(r"Frequency $\nu$  (increasing $\rightarrow$ shorter $\lambda$)", fontsize=12)
    ax.set_ylabel(r"Refractive Index $n(\nu)$", fontsize=12)
    ax.set_title(r"Refractive Index of a Medium with Two Narrow Resonances", fontsize=14, pad=10)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", fontsize=10)

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "q4_refractive_index_two_res.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_susceptibility()
    plot_two_resonance_index()
    print("All Sheet 1 plots generated.")
