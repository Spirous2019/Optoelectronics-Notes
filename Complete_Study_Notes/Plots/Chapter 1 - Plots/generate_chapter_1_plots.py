"""
generate_chapter_1_plots.py
===========================
Generates all matplotlib figures for Chapter 1 of the Optoelectronics notes.

Produces:
  - susceptibility_profiles.jpg         :  The Susceptibility Line Shapes (chi', chi'') vs normalised detuning δ.
  - output_transmission_spectrum.jpg    :  The Output Transmission Spectrum with two Lorentzian absorption dips.
  - lineshape_function_lorentzian.jpg   :  The normalised Lorentzian line shape function g(ν).
  - velocity_distribution.jpg           :  Maxwell-Boltzmann 1-D velocity PDF.
  - inhomogeneous_broadening.jpg        :  Shifted Lorentzians → Gaussian ensemble profile.
  - underdamped_oscillation.jpg         :  Underdamped harmonic oscillation with decay envelope.
  - three_damping_regimes.jpg           :  Under-, critically, and over-damped responses.
  - resonance_amplitude_phase.jpg       :  Driven oscillator amplitude & phase response (2-panel).
  - anomalous_dispersion_single.jpg     :  n(ω) near a single resonance: normal vs anomalous dispersion.
  - broadband_refractive_index.jpg      :  n(ω) across multiple resonances.
  - delta_vs_real_absorption.jpg        :  Ideal δ(ν−ν₀) vs realistic Lorentzian absorption.
  - energy_decay_profile.jpg            :  Exponential energy decay U(t) = U₀ e^{-t/τ_t}.
  - power_decay_lossy_medium.jpg        :  EM attenuation: E(z), amplitude envelope A(z), power P(z)∝|E|².

Run this script from any working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = SCRIPT_DIR   # all plots saved here for manual review

# ── Colour palette (figure-generation-style) ───────────────────────────────
WHITE    = "#ffffff"
AXES_CLR = "#2b2b2b"
GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"
CORAL    = "#d94452"
GOLD     = "#c28800"
LAVENDER = "#7e57c2"
SKYBLUE  = "#1976d2"
MINT     = "#00897b"
ORANGE   = "#e65100"
PINK     = "#ad1457"

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

    # ── Add TEAL dot at right FWHM endpoint ────────────────────────────────
    ax.plot(1.0, fwhm_y, 'o', color=TEAL, markersize=7, zorder=6)

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.85, 1.15)
    ax.set_xlabel(r"Normalised Detuning $\delta = 2(\omega - \omega_0)/\eta$", fontsize=13)
    ax.set_ylabel(r"Susceptibility Component (normalised)", fontsize=13)
    ax.set_title("The Fundamental Susceptibility Line Shapes", fontsize=15, pad=12)
    ax.grid(True)

    # x-axis: keep only the physically meaningful ticks (-1, 0, +1)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([r"$-1$", r"$0$", r"$1$"], fontsize=12)

    # y-axis: keep only the meaningful amplitude landmarks
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels([r"$-\chi'_{\max}$", r"$0$", r"$\chi'_{\max}$", r"$|\chi''|_{\max}$"], fontsize=11)

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

    # ── Legend (curve + FWHM span only — no shading entry) ─────────────────
    handles = [
        plt.Line2D([0], [0], color=TEAL, linewidth=2.5,
                   label=r"$g_{\nu_0}(\nu)$ — Lorentzian line shape"),
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
    bands entirely with shaded backgrounds.
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

    # ── Transmission curve ───────────────────────────────────────────────────
    ax.plot(nu, T, color=TEAL, linewidth=2.5, label="Transmission Spectrum")

    # ── FWHM Shading (drawn first so the curve sits on top) ─────────────────
    ax.axvspan(b1_lo, b1_hi, alpha=0.11, color=CORAL,
               label=r"Absorption Bandwidth ($\Delta\nu$ FWHM)")
    ax.axvspan(b2_lo, b2_hi, alpha=0.11, color=CORAL)

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels([r"$0$", r"$1$"], fontsize=11)
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

    # ── Label for the double-headed 2σ_V arrow ───────────────────────────────
    ax.text(0, f_sig + 0.016 * f_peak,
            r"$2\sigma_V$", fontsize=10, color=CORAL,
            ha="center", va="bottom")

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

    # ── L-shaped arrow: label closer to curve, shorter horizontal arm ─────────
    ax.annotate(
        r"$\Delta\nu_D = \sqrt{8\ln 2}\;\sigma_D$",
        xy=(nu0, g_half),
        xytext=(2.3 * sigma_D, g_half + 0.07 * g_peak),
        fontsize=9.5, color=CORAL, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5,
                        connectionstyle="angle,angleA=180,angleB=-90")
    )

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
    ax.set_ylim(0, g_peak * 1.55)   # extra headroom for annotation
    ax.set_xlabel(r"Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Spectral Intensity (normalised)", fontsize=13)
    ax.set_title(r"Inhomogeneous Broadening: Shifted Lorentzians and the Gaussian Ensemble Profile",
                 fontsize=15, pad=12)
    ax.grid(True)

    # ── Legend: Gaussian → Doppler FWHM → Individual atom → Individual peak ──
    handles = [
        plt.Line2D([0], [0], color=TEAL, linewidth=2.5,
                   label=r"Gaussian ensemble profile $\bar{g}(\nu)$"),
        plt.Line2D([0], [0], color=CORAL, linewidth=1.8,
                   label=r"Doppler FWHM  $(\Delta\nu_D = \sqrt{8\ln 2}\;\sigma_D)$"),
        plt.Line2D([0], [0], color=LAVENDER, linewidth=1.4, alpha=0.7,
                   label=r"Individual atom $g_{\nu_\mathrm{res}}(\nu)$"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LAVENDER,
                   markersize=7, label=r"Individual peak $g_{\nu_\mathrm{res}}(\nu_\mathrm{res})$"),
    ]
    ax.legend(handles=handles, loc="upper right",
              framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc",
              labelcolor=AXES_CLR, fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "inhomogeneous_broadening.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)



# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 6  —  Underdamped Harmonic Oscillation
# ═══════════════════════════════════════════════════════════════════════════

def plot_underdamped_oscillation():
    """x(t) = A·e^{-ηt/2}·cos(ω_d·t) with ± decay envelope."""
    eta = 2.0; omega_d = 20.0; A = 1.0
    t = np.linspace(0, 3.5, 8000)
    env_pos = A * np.exp(-eta * t / 2)
    env_neg = -env_pos
    x = A * np.exp(-eta * t / 2) * np.cos(omega_d * t)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t, x, color=SKYBLUE, lw=2.5, label=r"Oscillating Displacement $x(t)$")
    ax.plot(t, env_pos, color=CORAL, lw=2.5, ls="--",
            label=r"Decay Envelopes $\pm A e^{-\eta t/2}$")
    ax.plot(t, env_neg, color=CORAL, lw=2.5, ls="--")
    ax.axhline(0, color=AXES_CLR, lw=0.8)

    ax.set_xlim(0, 3.5); ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([r"$-A$", r"$0$", r"$A$"], fontsize=11)
    ax.set_xlabel(r"Time $t$", fontsize=13)
    ax.set_ylabel(r"Displacement $x(t)$", fontsize=13)
    ax.set_title("Underdamped Harmonic Oscillation", fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "underdamped_oscillation.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 7  —  Three Damping Regimes
# ═══════════════════════════════════════════════════════════════════════════

def plot_three_damping_regimes():
    """Underdamped, critically damped, overdamped on one plot."""
    omega0 = 10.0; x0 = 1.0
    t = np.linspace(0, 3.0, 8000)

    # Underdamped: gamma < omega0
    gamma_u = 3.0
    wd = np.sqrt(omega0**2 - gamma_u**2)
    x_under = x0 * np.exp(-gamma_u * t) * (
        np.cos(wd * t) + (gamma_u / wd) * np.sin(wd * t))

    # Critically damped: gamma = omega0
    gamma_c = omega0
    x_crit = x0 * (1 + gamma_c * t) * np.exp(-gamma_c * t)

    # Overdamped: gamma > omega0
    gamma_o = 15.0
    s1 = -gamma_o + np.sqrt(gamma_o**2 - omega0**2)
    s2 = -gamma_o - np.sqrt(gamma_o**2 - omega0**2)
    C1 = x0 * s2 / (s2 - s1); C2 = -x0 * s1 / (s2 - s1)
    x_over = C1 * np.exp(s1 * t) + C2 * np.exp(s2 * t)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t, x_under, color=SKYBLUE, lw=2.5, ls="-",
            label="Underdamped: Oscillatory")
    ax.plot(t, x_crit, color=TEAL, lw=2.5, ls="--",
            label="Critically Damped: Fastest Return")
    ax.plot(t, x_over, color=CORAL, lw=2.5, ls=":",
            label="Overdamped: Sluggish Exponential")
    ax.axhline(0, color=AXES_CLR, lw=0.8)

    ax.set_xlim(0, 3.0); ax.set_ylim(-0.45, 1.1)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$0$", r"$x_0$"], fontsize=11)
    ax.set_xlabel(r"Time $t$", fontsize=13)
    ax.set_ylabel(r"Displacement $x(t)$", fontsize=13)
    ax.set_title("The Three Damping Regimes of the Classical Oscillator",
                 fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "three_damping_regimes.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 8  —  Resonance: Amplitude Divergence and Phase Transition
# ═══════════════════════════════════════════════════════════════════════════

def plot_resonance_amplitude_phase():
    """2-panel: amplitude |x0| and phase phi vs driving frequency."""
    omega0 = 1.0; gamma = 0.04; F0 = 1.0
    omega = np.linspace(0.01, 2.0, 8000)

    amp = F0 / np.sqrt((omega0**2 - omega**2)**2 + (2 * gamma * omega)**2)
    phase_rad = -np.arctan2(2 * gamma * omega, omega0**2 - omega**2)
    phase_deg = np.degrees(phase_rad)

    omega_lo = omega0 - gamma; omega_hi = omega0 + gamma

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Top: Amplitude (no FWHM shading)
    ax1.plot(omega, amp, color=SKYBLUE, lw=2.5, label="Normalized Amplitude")
    ax1.axvline(omega0, color=AXES_CLR, lw=0.9, ls="--")
    ax1.set_ylabel(r"Amplitude Response $|x_0|$", fontsize=13)
    ax1.set_ylim(0, amp.max() * 1.12)
    ax1.set_yticks([0, amp.max()])
    ax1.set_yticklabels([r"$0$", r"$|x_0|_{\max}$"], fontsize=11)
    ax1.set_title("Resonance: Amplitude Divergence and Phase Transition",
                  fontsize=15, pad=12)
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
               edgecolor="#cccccc", labelcolor=AXES_CLR)

    # Bottom: Phase
    ax2.plot(omega, phase_deg, color=CORAL, lw=2.5,
             label=r"Phase Shift $\phi$")
    ax2.axvline(omega0, color=AXES_CLR, lw=0.9, ls="--")
    ax2.axhline(-90, color=LAVENDER, lw=1.2, ls="--")
    ax2.annotate(r"Crosses at exactly $-90^\circ$",
                 xy=(omega0, -90), xytext=(omega0 + 0.25, -73),
                 fontsize=10, color=CORAL, ha="left",
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.3,
                                 connectionstyle="arc3,rad=0"))
    ax2.set_ylabel(r"Phase Shift $\phi$ (Degrees)", fontsize=13)
    ax2.set_ylim(-195, 15)
    ax2.set_yticks([0, -90, -180])
    ax2.set_yticklabels([r"$0^\circ$", r"$-90^\circ$", r"$-180^\circ$"])
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
               edgecolor="#cccccc", labelcolor=AXES_CLR)
    ax2.set_xlabel(r"Driving Frequency $\omega$", fontsize=13)
    ax2.set_xlim(0.01, 2.0)
    ax2.set_xticks([omega0])
    ax2.set_xticklabels([r"$\omega_0$"])

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "resonance_amplitude_phase.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 9  —  Anomalous Dispersion (Single Resonance)
# ═══════════════════════════════════════════════════════════════════════════

def plot_anomalous_dispersion_single():
    """n(omega) near a single resonance: normal/anomalous bands shaded."""
    omega0 = 1.0; gamma = 0.06; wp2 = 0.08
    omega = np.linspace(0.01, 2.0, 10000)

    denom = (omega0**2 - omega**2)**2 + (gamma * omega)**2
    n_sq = 1.0 + wp2 * (omega0**2 - omega**2) / denom
    n_sq = np.clip(n_sq, 0.01, None)
    n = np.sqrt(n_sq)

    # Find the exact local extrema to mathematically bound the regions
    idx_max = np.argmax(n)
    idx_min = np.argmin(n)
    w_max = omega[idx_max]
    w_min = omega[idx_min]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade Normal Dispersion regions (dn/dω > 0)
    ax.axvspan(0.01, w_max, alpha=0.08, color=TEAL)
    ax.axvspan(w_min, 2.0, alpha=0.08, color=TEAL)

    # Shade Anomalous Dispersion region (dn/dω < 0)
    ax.axvspan(w_max, w_min, alpha=0.15, color=PINK)

    ax.plot(omega, n, color=SKYBLUE, lw=2.5,
            label=r"Refractive Index $n(\omega)$")
    ax.axhline(1.0, color=AXES_CLR, lw=0.8, ls="--")

    ax.set_xlim(0.01, 2.0)
    padding = (n.max() - n.min()) * 0.12
    ax.set_ylim(n.min() - padding, n.max() + padding)

    # Remove numbers on y-axis to make it general
    ax.set_yticks([])
    ax.set_xticks([omega0])
    ax.set_xticklabels([r"Resonance $\omega_0$"], fontsize=12)

    ax.set_xlabel(r"Driving Frequency $\omega$", fontsize=13)
    ax.set_ylabel(r"Refractive Index $n$", fontsize=13)
    ax.set_title("Normal vs Anomalous Dispersion (Single Resonance)",
                 fontsize=15, pad=12)
    ax.grid(True)

    # Create descriptive legend handles mapping the shaded regions
    import matplotlib.patches as mpatches
    handles = [
        plt.Line2D([0], [0], color=SKYBLUE, lw=2.5,
                   label=r"Refractive Index $n(\omega)$"),
        mpatches.Patch(facecolor=TEAL, alpha=0.25,
                       label="Normal Dispersion"),
        mpatches.Patch(facecolor=PINK, alpha=0.35,
                       label="Anomalous Dispersion"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "anomalous_dispersion_single.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 10  —  Broadband Refractive Index (Multiple Resonances)
# ═══════════════════════════════════════════════════════════════════════════

def plot_broadband_refractive_index():
    """n(omega) across 3 resonances with normal/anomalous bands shaded."""
    resonances = [(1.0, 0.08, 0.07), (2.5, 0.08, 0.12), (4.0, 0.08, 0.16)]
    omega = np.linspace(0.01, 5.5, 20000)

    n_sq = np.ones_like(omega)
    for om0, gam, wp2 in resonances:
        denom = (om0**2 - omega**2)**2 + (gam * omega)**2
        n_sq += wp2 * (om0**2 - omega**2) / denom
    n_sq = np.clip(n_sq, 0.60, None)   # keep n physical; first resonance deepest
    n = np.sqrt(n_sq)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Mathematically locate boundaries using dn/dω sign crossings
    dn = np.gradient(n, omega)
    crossings = np.where(np.diff(np.sign(dn)) != 0)[0]
    bounds = [0] + list(crossings) + [len(omega) - 1]

    for i in range(len(bounds) - 1):
        idx1, idx2 = bounds[i], bounds[i+1]
        mid_idx = (idx1 + idx2) // 2
        if dn[mid_idx] > 0:
            ax.axvspan(omega[idx1], omega[idx2], alpha=0.08, color=TEAL)
        else:
            ax.axvspan(omega[idx1], omega[idx2], alpha=0.15, color=PINK)

    ax.plot(omega, n, color=SKYBLUE, lw=2.5,
            label=r"Refractive Index $n(\omega)$")
    ax.axhline(1.0, color=AXES_CLR, lw=0.8, ls="--")

    ax.set_xticks([r[0] for r in resonances])
    ax.set_xticklabels(
        [rf"$\omega_{{0,\,{i+1}}}$" for i in range(len(resonances))],
        fontsize=12)

    ax.set_xlim(0.01, 5.5)
    ax.set_ylim(0.74, 1.28)
    ax.set_yticks([1.0])
    ax.set_yticklabels([r"$1$"], fontsize=12)

    ax.set_xlabel(r"Frequency $\omega$", fontsize=13)
    ax.set_ylabel(r"Refractive Index $n$", fontsize=13)
    ax.set_title("Broadband Refractive Index across Multiple Resonances",
                 fontsize=15, pad=12)
    ax.grid(True)

    import matplotlib.patches as mpatches
    handles = [
        plt.Line2D([0], [0], color=SKYBLUE, lw=2.5,
                   label=r"Refractive Index $n(\omega)$"),
        mpatches.Patch(facecolor=TEAL, alpha=0.25,
                       label="Normal Dispersion"),
        mpatches.Patch(facecolor=PINK, alpha=0.35,
                       label="Anomalous Dispersion"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "broadband_refractive_index.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 11  —  Delta Function vs Real Smeared Absorption
# ═══════════════════════════════════════════════════════════════════════════

def plot_delta_vs_real_absorption():
    """Ideal delta(nu - nu0) spike vs realistic Lorentzian line."""
    nu0 = 5.0; dnu = 1.2
    nu = np.linspace(0, 10, 6000)
    g = (dnu / (2 * np.pi)) / ((dnu / 2)**2 + (nu - nu0)**2)
    g_peak = (dnu / (2 * np.pi)) / (dnu / 2)**2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(nu, g, color=SKYBLUE, lw=2.5,
            label="Real Absorption (Lorentzian)")

    delta_height = g_peak * 1.35
    ax.plot([nu0, nu0], [0, delta_height], color=CORAL, lw=2.5)
    ax.plot(nu0, delta_height, marker="^", color=CORAL, ms=10, zorder=5)
    # Label placed directly to the left of the spike tip — no arrow
    ax.text(nu0 - 0.15, delta_height * 0.97,
            r"Ideal $\delta(\nu - \nu_0)$" + "\n(Infinitely Sharp)",
            fontsize=10, color=CORAL, ha="right", va="top")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, delta_height * 1.15)   # y origin at zero
    ax.set_yticks([])                       # general — no numeric y values
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=12)
    ax.set_xlabel(r"Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Absorption Profile $g(\nu)$", fontsize=13)
    ax.set_title("Delta Function Idealization vs Real Smeared Absorption",
                 fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "delta_vs_real_absorption.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 12  —  Exponential Energy Decay with Total Lifetime
# ═══════════════════════════════════════════════════════════════════════════

def plot_energy_decay_profile():
    """U(t) = U0 * exp(-t / tau_t) with tau_t crosshairs."""
    U0 = 1.0; tau_t = 10.0
    t = np.linspace(0, 45, 4000)
    U = U0 * np.exp(-t / tau_t)
    U_at_tau = U0 * np.exp(-1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, U, color=TEAL, lw=2.5,
            label=r"Energy $U(t) = U_0 e^{-t/\tau_t}$")

    # Dashed crosshairs to the 1/e point
    ax.plot([0, tau_t], [U_at_tau, U_at_tau],
            color=LAVENDER, lw=1.5, ls="--")
    ax.plot([tau_t, tau_t], [0, U_at_tau],
            color=LAVENDER, lw=1.5, ls="--")
    ax.plot(tau_t, U_at_tau, "o", color=LAVENDER, ms=7, zorder=5)

    # U_0/e as a y-axis tick label, tau_t as an x-axis tick label
    ax.set_yticks([U0, U_at_tau])
    ax.set_yticklabels([r"$U_0$", r"$U_0/e$"], fontsize=11)
    ax.get_yticklabels()[1].set_color(LAVENDER)
    ax.set_xticks([tau_t])
    ax.set_xticklabels([r"$\tau_t$"], fontsize=11, color=LAVENDER)

    # "Total Lifetime" label placed near the dot — no arrow
    ax.text(tau_t + 1.2, U_at_tau + 0.04,
            r"Total Lifetime $\tau_t$",
            fontsize=11, color=LAVENDER, ha="left", va="bottom")

    ax.set_xlim(0, 46)    # x origin at zero
    ax.set_ylim(0, 1.08)  # y origin at zero
    ax.set_xlabel(r"Time $t$", fontsize=13)
    ax.set_ylabel(r"Energy $U(t)$", fontsize=13)
    ax.set_title("Exponential Energy Decay with Total Lifetime",
                 fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "energy_decay_profile.jpg")
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
    plot_underdamped_oscillation()
    plot_three_damping_regimes()
    plot_resonance_amplitude_phase()
    plot_anomalous_dispersion_single()
    plot_broadband_refractive_index()
    plot_delta_vs_real_absorption()
    plot_energy_decay_profile()
