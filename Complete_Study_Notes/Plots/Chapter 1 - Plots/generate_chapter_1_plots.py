"""
generate_chapter_1_plots.py
===========================
Generates all matplotlib figures for Chapter 1 of the Optoelectronics notes.

Currently produces:
  - susceptibility_profiles.jpg         :  The Susceptibility Line Shapes (chi', chi'') vs normalised detuning δ.
  - output_transmission_spectrum.jpg    :  The Output Transmission Spectrum with two Lorentzian absorption dips.
  - lineshape_function_lorentzian.jpg   :  The normalised Lorentzian line shape function g(ν), with the peak
                                           marked by a dot+label, the FWHM indicated by a double-headed arrow,
                                           and all key x/y axis values labelled.

Run this script from any working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = SCRIPT_DIR   # plots saved next to the script

# Output directory for the Figures folder (lineshape figure saved there directly)
FIGURES_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "Figures", "Chapter 1")
)

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
#  FIGURE 1  —  Susceptibility profiles χ'(δ) and χ''(δ)
# ═══════════════════════════════════════════════════════════════════════════

def plot_susceptibility_lineshapes():
    """
    χ'(δ)  = -δ / (1 + δ²)     [dispersive / real part]
    χ''(δ) = -1 / (1 + δ²)     [absorptive / imaginary part]
    """

    δ = np.linspace(-5, 5, 4000)

    def calc_chi_prime(d): return -d / (1 + d**2)
    def calc_chi_dprime(d): return 1.0 / (1 + d**2)

    chi_prime = calc_chi_prime(δ)
    chi_dprime = calc_chi_dprime(δ)

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
    # We strictly calculate coordinates dynamically via the math functions.

    # Absorption peak
    p_abs = calc_chi_dprime(0.0)
    ax.plot(0, p_abs, 'o', color=TEAL, markersize=7)
    ax.annotate(r"$|\chi''|_{\max}$", xy=(0, p_abs), xytext=(-0.5, p_abs + 0.08),
                fontsize=12, color=TEAL, ha="center", va="center")

    # Dispersion positive peak
    p_disp_pos = calc_chi_prime(-1.0)
    ax.plot(-1.0, p_disp_pos, 'o', color=CORAL, markersize=7)
    ax.annotate(r"$\chi'_{\max}$", xy=(-1.0, p_disp_pos), xytext=(-1.6, p_disp_pos + 0.1),
                fontsize=12, color=CORAL, ha="center", va="center")

    # Dispersion negative peak
    p_disp_neg = calc_chi_prime(1.0)
    ax.plot(1.0, p_disp_neg, 'o', color=CORAL, markersize=7)
    ax.annotate(r"$-\chi'_{\max}$", xy=(1.0, p_disp_neg), xytext=(1.6, p_disp_neg - 0.1),
                fontsize=12, color=CORAL, ha="center", va="center")

    # ── FWHM double-headed arrow on χ'' ─────────────────────────────────────
    fwhm_y = calc_chi_dprime(1.0)
    ax.annotate("", xy=(1.0, fwhm_y), xytext=(-1.0, fwhm_y),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.8))
    ax.text(0, fwhm_y + 0.04, r"FWHM", fontsize=9, color=TEAL, ha="center", va="bottom")

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.85, 1.15)
    ax.set_xlabel(r"Normalised Detuning $\delta = 2(\omega - \omega_0)/\eta$", fontsize=13)
    ax.set_ylabel(r"Susceptibility Component (normalised)", fontsize=13)
    ax.set_title("The Fundamental Susceptibility Line Shapes", fontsize=15, pad=12)
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    ax.set_xticklabels([r"$-5$", r"$-4$", r"$-3$", r"$-2$", r"$-1$",
                         r"$0$",  r"$1$",  r"$2$",  r"$3$",  r"$4$",  r"$5$"])

    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "susceptibility_profiles.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3  —  Normalised Lorentzian Line Shape Function g_ν₀(ν)
# ═══════════════════════════════════════════════════════════════════════════

def plot_lineshape_lorentzian():
    """
    g(ν) = (Δν/2π) / [ (Δν/2)² + (ν − ν₀)²  ]

    Normalised Lorentzian line shape function.  The figure shows:
      • The Lorentzian curve in TEAL.
      • FWHM region shaded in TEAL down to the x-axis.
      • A filled dot at the peak (ν₀, g_max) in TEAL with a matching text label.
      • A ←→ double-headed arrow spanning the FWHM at height g_max/2.
      • Dots at the two FWHM boundary points in CORAL.
      • x-axis ticks: ν₀−Δν/2, ν₀, ν₀+Δν/2.
      • y-axis ticks: 0, g_max/2, g_max.
      • A legend.
    """

    # ── Physical parameters (arbitrary normalised units) ─────────────────────
    nu0  = 0.0    # resonance centre
    dnu  = 1.0    # FWHM  (sets the scale; labels are written symbolically)

    # ── Derived quantities ───────────────────────────────────────────────────
    g_max  = 2.0 / (np.pi * dnu)   # peak amplitude  =  2/(π Δν)
    g_half = g_max / 2.0           # half-maximum height

    # ── Frequency axis ───────────────────────────────────────────────────────
    nu = np.linspace(nu0 - 2.5 * dnu, nu0 + 2.5 * dnu, 6000)

    def lorentzian(f):
        return (dnu / (2 * np.pi)) / ((dnu / 2)**2 + (f - nu0)**2)

    g = lorentzian(nu)

    # ── FWHM boundary frequencies ────────────────────────────────────────────
    nu_left  = nu0 - dnu / 2
    nu_right = nu0 + dnu / 2

    # Mask for the FWHM region only
    fwhm_mask = (nu >= nu_left) & (nu <= nu_right)

    # ── Figure / axes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    # ── FWHM shading (down to x-axis, drawn first so curve sits on top) ──────
    ax.fill_between(nu, g, where=fwhm_mask, alpha=0.18, color=TEAL,
                    label=r"FWHM region")

    # ── Main Lorentzian curve ────────────────────────────────────────────────
    ax.plot(nu, g, color=TEAL, linewidth=2.5, label=r"$g_{\nu_0}(\nu)$")

    # ── Peak dot + label (both in TEAL) ──────────────────────────────────────
    ax.plot(nu0, g_max, 'o', color=TEAL, markersize=8, zorder=5,
            label=r"$g_{\max} = \dfrac{2}{\pi\,\Delta\nu}$")
    ax.text(nu0 + 0.09 * dnu, g_max * 1.025,
            r"$g_{\max} = \dfrac{2}{\pi\,\Delta\nu}$",
            fontsize=11, color=TEAL, ha="left", va="bottom")

    # ── FWHM double-headed arrow ──────────────────────────────────────────────
    ax.annotate("",
                xy=(nu_right, g_half), xytext=(nu_left, g_half),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.8))
    ax.text(nu0, g_half + 0.022 * g_max,
            r"$\Delta\nu$  (FWHM)",
            fontsize=10, color=CORAL, ha="center", va="bottom",
            label=r"$\Delta\nu$ (FWHM)")

    # ── Dots at FWHM boundary points ─────────────────────────────────────────
    ax.plot([nu_left, nu_right], [g_half, g_half],
            'o', color=CORAL, markersize=6, zorder=5,
            label=r"Half-maximum points  $\left(\nu_0 \pm \tfrac{\Delta\nu}{2},\;\dfrac{g_{\max}}{2}\right)$")

    # ── x-axis ticks: ν₀−Δν/2, ν₀, ν₀+Δν/2 ─────────────────────────────────
    x_ticks = [nu_left, nu0, nu_right]
    x_labels = [
        r"$\nu_0 - \frac{\Delta\nu}{2}$",
        r"$\nu_0$",
        r"$\nu_0 + \frac{\Delta\nu}{2}$",
    ]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=11)

    # ── y-axis ticks: 0, g_max/2, g_max ──────────────────────────────────────
    y_ticks  = [0, g_half, g_max]
    y_labels = [
        r"$0$",
        r"$\dfrac{g_{\max}}{2} = \dfrac{1}{\pi\,\Delta\nu}$",
        r"$g_{\max} = \dfrac{2}{\pi\,\Delta\nu}$",
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)

    # ── Axis limits, labels, title ────────────────────────────────────────────
    ax.set_xlim(nu[0], nu[-1])
    ax.set_ylim(0, g_max * 1.28)          # starts exactly at 0 so shading touches x-axis
    ax.set_xlabel(r"Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"$g_{\nu_0}(\nu)$  [Hz$^{-1}$]", fontsize=13)
    ax.set_title(r"The Normalised Lorentzian Line Shape Function", fontsize=15, pad=12)
    ax.grid(True)

    # ── Legend (curve + FWHM region + FWHM span only) ────────────────────────
    handles = [
        plt.Line2D([0], [0], color=TEAL, linewidth=2.5,
                   label=r"$g_{\nu_0}(\nu)$ — Lorentzian line shape"),
        plt.matplotlib.patches.Patch(facecolor=TEAL, alpha=0.35,
                   label=r"FWHM region  ($\nu_0 \pm \Delta\nu/2$)"),
        plt.Line2D([0], [0], color=CORAL, linewidth=1.8,
                   label=r"$\Delta\nu$ (FWHM)"),
    ]
    ax.legend(handles=handles, loc="upper right",
              framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc",
              labelcolor=AXES_CLR, fontsize=10)

    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "lineshape_function_lorentzian.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2  —  Output Transmission Spectrum
# ═══════════════════════════════════════════════════════════════════════════

def plot_transmission_spectrum():
    """
    T(ν) = exp(−α₁(ν) − α₂(ν))  where αᵢ(ν) is a Lorentzian.
    Demonstrates "Shading First, Labels Second" rule by conveying the FWHM
    bands and transparency window entirely with shaded backgrounds.
    """
    nu = np.linspace(0.0, 11.0, 6000)

    # ── Resonance parameters ─────────────────────────────────────────────────
    nu0_1, dnu_1, aL_1 = 3.0, 0.50, 1.50   # ν₀, FWHM, α_peak·L
    nu0_2, dnu_2, aL_2 = 7.5, 0.70, 2.50

    def calc_alpha(nu_val, nu0, dnu, aL):
        return aL / (1.0 + ((nu_val - nu0) / (dnu / 2.0))**2)

    alpha_1 = calc_alpha(nu, nu0_1, dnu_1, aL_1)
    alpha_2 = calc_alpha(nu, nu0_2, dnu_2, aL_2)
    T = np.exp(-(alpha_1 + alpha_2))

    # ── FWHM band boundaries (calculated directly using math) ──────────────
    b1_lo, b1_hi = nu0_1 - dnu_1 / 2, nu0_1 + dnu_1 / 2
    b2_lo, b2_hi = nu0_2 - dnu_2 / 2, nu0_2 + dnu_2 / 2

    # ── Figure setup ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    # ── FWHM Shading (drawn first so the curve sits on top) ─────────────────
    ax.axvspan(b1_lo, b1_hi, alpha=0.11, color=CORAL,
               label=r"Absorption Bandwidth ($\Delta\nu$ FWHM)")
    ax.axvspan(b2_lo, b2_hi, alpha=0.11, color=CORAL)

    # ── Transparency window shading (subtle definition of the gap region) ────
    # Boundaries are exactly b1_hi and b2_lo so green starts precisely where red ends.
    ax.axvspan(b1_hi, b2_lo, alpha=0.06, color=TEAL,
               label="Transparency Window")

    # ── Transmission curve ───────────────────────────────────────────────────
    ax.plot(nu, T, color=TEAL, linewidth=2.5, label="Transmission Spectrum")

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks([nu0_1, nu0_2])
    ax.set_xticklabels([r"$\nu_{0,1}$", r"$\nu_{0,2}$"], fontsize=12)
    ax.set_xlabel(r"Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Output Transmission $T(\nu) = e^{-\alpha(\nu)L}$", fontsize=13)
    ax.set_title("The Output Transmission Spectrum", fontsize=15, pad=12)
    ax.grid(True)

    # Place legend clearly inside a white region
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "output_transmission_spectrum.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)



# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 4  —  Maxwell-Boltzmann Velocity Distribution  f_V(V)
# ═══════════════════════════════════════════════════════════════════════════

MINT   = "#00897b"
ORANGE = "#e65100"

def plot_velocity_distribution():
    """
    f_V(V) = 1 / (sqrt(2π) σ_V) · exp(−V² / 2σ_V²)

    Single panel showing the Maxwell-Boltzmann 1-D velocity PDF.
    Features:
      • Curve in TEAL.
      • Dot at the peak (V=0, f_max) in TEAL with centered label.
      • Dots at V = ±σ_V on the curve in CORAL with individual labels.
      • No shading or dotted lines.
    """
    sigma_V = 1.0          # normalised; axis labels are written symbolically
    V       = np.linspace(-4.0 * sigma_V, 4.0 * sigma_V, 6000)

    def fV(v):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma_V)) * np.exp(-v**2 / (2 * sigma_V**2))

    f      = fV(V)
    f_peak = fV(0.0)                  # maximum value at V = 0
    f_sig  = fV(sigma_V)              # value at V = ±σ_V  (= f_peak / sqrt(e))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    # ── Main PDF curve ───────────────────────────────────────────────────────
    ax.plot(V, f, color=TEAL, linewidth=2.5, label=r"$f_V(V)$")

    # ── Dot at peak (V = 0) ──────────────────────────────────────────────────
    ax.plot(0, f_peak, 'o', color=TEAL, markersize=8, zorder=5)
    ax.text(0, f_peak * 1.04,
            r"$f_{\max} = \dfrac{1}{\sqrt{2\pi}\,\sigma_V}$",
            fontsize=11, color=TEAL, ha="center", va="bottom")

    # ── Standard deviation spanning arrow ────────────────────────────────────
    ax.plot([-sigma_V, sigma_V], [f_sig, f_sig], 'o', color=CORAL, markersize=6, zorder=7)
    ax.annotate("",
                xy=(sigma_V, f_sig), xytext=(-sigma_V, f_sig),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.8))

    # ── x-axis ticks: −2σ, −σ, 0, +σ, +2σ ──────────────────────────────────
    x_ticks  = [-2*sigma_V, -sigma_V, 0, sigma_V, 2*sigma_V]
    x_labels = [r"$-2\sigma_V$", r"$-\sigma_V$", r"$0$",
                r"$\sigma_V$",   r"$2\sigma_V$"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=11)

    # ── y-axis ticks: 0, f(σ), f_peak ───────────────────────────────────────
    ax.set_yticks([0, f_sig, f_peak])
    ax.set_yticklabels([r"$0$",
                        r"$f_{\max} e^{-1/2}$",
                        r"$f_{\max}$"], fontsize=11)

    # ── Axis limits, labels, title ───────────────────────────────────────────
    ax.set_xlim(V[0], V[-1])
    ax.set_ylim(0, f_peak * 1.32)
    ax.set_xlabel(r"Velocity $V$ along propagation axis", fontsize=13)
    ax.set_ylabel(r"Probability Density $f_V(V)$", fontsize=13)
    ax.set_title(r"Maxwell-Boltzmann Single-Axis Velocity Distribution", fontsize=15, pad=12)
    ax.grid(True)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=TEAL,  linewidth=2.5, label=r"Maxwell-Boltzmann PDF $f_V(V)$"),
        plt.Line2D([0], [0], color=CORAL, linewidth=1.8, label=r"Standard Deviation spread ($2\sigma_V$)"),
    ]
    ax.legend(handles=handles, loc="upper right",
              framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc",
              labelcolor=AXES_CLR, fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "velocity_distribution.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 5  —  Inhomogeneous Broadening: Shifted Lorentzians → Gaussian
# ═══════════════════════════════════════════════════════════════════════════

LAVENDER = "#7e57c2"

def plot_inhomogeneous_broadening():
    """
    Illustrates inhomogeneous Doppler broadening:
      • Several individual atom Lorentzians weighted by the Gaussian PDF, in LAVENDER.
      • Dots at every individual peak, tracing the Gaussian envelope shape.
      • The Gaussian ensemble profile ḡ(ν) in TEAL on top.
      • Dot at the Gaussian peak (ν₀, ḡ_max) in TEAL with centred label using σ_D.
      • FWHM shading on the Gaussian, with Δν_D = 2√(2 ln2)·σ_D double-headed arrow.
      • Half-maximum point marked in CORAL.
    """
    # ── Parameters ───────────────────────────────────────────────────────────
    nu0    = 0.0
    sigma_D = 1.0       # Doppler broadening parameter (std dev in frequency units)
    dnu    = 0.18       # individual Lorentzian FWHM  (≪ σ_D → inhomogeneous limit)

    # Doppler FWHM of the Gaussian envelope
    dnu_D = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_D

    n_atoms = 9
    shifts  = np.linspace(-2.5 * sigma_D, 2.5 * sigma_D, n_atoms)

    nu = np.linspace(-5.0 * sigma_D, 5.0 * sigma_D, 8000)

    def lorentzian(f, nu_res):
        return (dnu / (2 * np.pi)) / ((dnu / 2)**2 + (f - nu_res)**2)

    def gaussian_envelope(f):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma_D)) * np.exp(-f**2 / (2 * sigma_D**2))

    g_gauss = gaussian_envelope(nu)
    g_peak  = gaussian_envelope(nu0)        # = 1 / (√(2π) σ_D)
    g_half  = g_peak / 2.0                  # half-maximum of Gaussian

    # FWHM boundary frequencies
    nu_left  = nu0 - dnu_D / 2
    nu_right = nu0 + dnu_D / 2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)



    # ── Individual shifted Lorentzians (weighted by Gaussian PDF) ───────────
    peak_xs, peak_ys = [], []
    for i, nu_res in enumerate(shifts):
        L        = lorentzian(nu, nu_res)
        weight   = gaussian_envelope(nu_res)
        L_scaled = L / np.max(L) * weight          # peak height == g(ν_res)
        label    = r"Individual atom $g_{\nu_\mathrm{res}}(\nu)$" if i == 0 else None
        ax.plot(nu, L_scaled, color=LAVENDER, linewidth=1.4,
                alpha=0.60, linestyle="-", label=label)
        peak_xs.append(nu_res)
        peak_ys.append(np.max(L_scaled))

    # ── Gaussian ensemble profile curve (on top) ─────────────────────────────
    ax.plot(nu, g_gauss, color=TEAL, linewidth=2.5,
            label=r"Gaussian ensemble profile $\bar{g}(\nu)$")

    # ── Dots at ALL individual Lorentzian peaks ───────────────────────────────
    ax.plot(peak_xs, peak_ys, 'o', color=LAVENDER, markersize=6,
            zorder=6, label=r"Individual peak $g_{\nu_\mathrm{res}}(\nu_\mathrm{res})$")

    # ── Dot at Gaussian peak with correct σ_D label ───────────────────────────
    ax.plot(nu0, g_peak, 'o', color=TEAL, markersize=8, zorder=7)
    ax.text(nu0, g_peak * 1.07,
            r"$\bar{g}_{\max} = \dfrac{1}{\sqrt{2\pi}\,\sigma_D}$",
            fontsize=11, color=TEAL, ha="center", va="bottom")

    # ── Half-maximum dots at FWHM boundary in CORAL ───────────────────────────
    ax.plot([nu_left, nu_right], [g_half, g_half],
            'o', color=CORAL, markersize=6, zorder=7)

    # ── FWHM double-headed arrow at g_half ────────────────────────────────────
    ax.annotate("",
                xy=(nu_right, g_half), xytext=(nu_left, g_half),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.8))

    # ── x-axis ticks ─────────────────────────────────────────────────────────
    x_ticks  = [-2*sigma_D, -sigma_D, nu0, sigma_D, 2*sigma_D]
    x_labels = [r"$\nu_0 - 2\sigma_D$", r"$\nu_0 - \sigma_D$",
                r"$\nu_0$",
                r"$\nu_0 + \sigma_D$", r"$\nu_0 + 2\sigma_D$"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)

    # ── y-axis ticks: 0, g_half, g_peak ──────────────────────────────────────
    ax.set_yticks([0, g_half, g_peak])
    ax.set_yticklabels([
        r"$0$",
        r"$\dfrac{\bar{g}_{\max}}{2}$",
        r"$\bar{g}_{\max} = \dfrac{1}{\sqrt{2\pi}\,\sigma_D}$",
    ], fontsize=10)

    ax.set_xlim(nu[0], nu[-1])
    ax.set_ylim(0, g_peak * 1.50)
    ax.set_xlabel(r"Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Spectral Intensity (normalised)", fontsize=13)
    ax.set_title(r"Inhomogeneous Broadening: Shifted Lorentzians and the Gaussian Ensemble Profile",
                 fontsize=15, pad=12)
    ax.grid(True)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=TEAL,     linewidth=2.5,
                   label=r"Gaussian ensemble profile $\bar{g}(\nu)$"),

        plt.Line2D([0], [0], color=LAVENDER, linewidth=1.4, alpha=0.7,
                   label=r"Individual atom $g_{\nu_\mathrm{res}}(\nu)$"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LAVENDER,
                   markersize=7, label=r"Individual peak $g_{\nu_\mathrm{res}}(\nu_\mathrm{res})$"),
        plt.Line2D([0], [0], color=CORAL, linewidth=1.8,
                   label=r"$\Delta\nu_D = \sqrt{8\ln 2}\;\sigma_D$  (Doppler FWHM)"),
    ]
    ax.legend(handles=handles, loc="upper right",
              framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc",
              labelcolor=AXES_CLR, fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "inhomogeneous_broadening.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)



# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_susceptibility_lineshapes()
    plot_transmission_spectrum()
    plot_lineshape_lorentzian()
    plot_velocity_distribution()
    plot_inhomogeneous_broadening()

