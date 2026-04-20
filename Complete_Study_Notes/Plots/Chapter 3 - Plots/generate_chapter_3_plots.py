"""
generate_chapter_3_plots.py
===========================
Generates matplotlib figures for Chapter 3 of the Optoelectronics notes.

Currently produces:
  - gain_compression.jpg : Normalised gain G/G_0 vs normalised input power P_in/P_s
                           for three small-signal gain values (10 dB, 20 dB, 30 dB),
                           showing the transition from the unsaturated regime to
                           near-unity gain in deep saturation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Image saved next to the script for manual review before moving to Figures/Chapter 3/
FIG_DIR = SCRIPT_DIR

# ── Colour palette ───────────────────────────────────────────────────────────
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

# ── Global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "axes.facecolor":   WHITE,
    "figure.facecolor": WHITE,
    "axes.edgecolor":   AXES_CLR,
    "axes.labelcolor":  AXES_CLR,
    "xtick.color":      AXES_CLR,
    "ytick.color":      AXES_CLR,
    "text.color":       AXES_CLR,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "lines.linewidth":  2.5,
    "grid.linestyle":   ":",
    "grid.linewidth":   0.6,
    "grid.alpha":       0.7,
    "grid.color":       GRID_CLR,
})


# ── Helper: implicit saturated-gain solver ───────────────────────────────────
def saturated_gain_curve(G0, pin_ps_array):
    """
    Solves G = G0 * exp(-(P_in/P_s) * (G - 1)) for G at each point in
    pin_ps_array using Brent's root-finding method.
    The solution is unique in [1, G0] for all P_in/P_s >= 0.
    """
    G = np.empty_like(pin_ps_array)
    for i, x in enumerate(pin_ps_array):
        def f(g):
            return g - G0 * np.exp(-x * (g - 1.0))
        # For x very close to 0, G ≈ G0; brentq handles the bracket safely
        try:
            G[i] = brentq(f, 1.0, G0 + 0.1)
        except ValueError:
            G[i] = G0
    return G


# ── Plot: Gain Compression ───────────────────────────────────────────────────
def plot_gain_compression():
    G0 = 10                           # Gives a smooth, wide compression curve
    x  = np.logspace(-3, 3, 1500)    # P_in / P_s: 0.001 → 1000
    G  = saturated_gain_curve(G0, x)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # ── Single gain curve (absolute G, not normalised) ────────────────────────
    ax.semilogx(x, G, color=TEAL, linewidth=2.5, label=r"Saturated Gain $G$")

    # ── Regime shading — named patches for the legend ─────────────────────────
    # Compute boundaries numerically from the curve:
    #   Small-signal: G ≥ 0.95 * G0 (within 5% of the unsaturated peak)
    #   Saturation:   G ≤ 1.05       (within 5% of absolute unity)
    import matplotlib.patches as mpatches

    x_ss_end  = x[np.where(G >= 0.95 * G0)[0][-1]]   # last x where G ≥ 0.95 G0
    x_sat_beg = x[np.where(G <= 1.05)[0][0]]          # first x where G ≤ 1.05

    ax.axvspan(x[0],    x_ss_end,  color=TEAL,  alpha=0.10)
    ax.axvspan(x_sat_beg, x[-1],   color=CORAL, alpha=0.10)

    small_signal_patch = mpatches.Patch(
        facecolor=TEAL,  alpha=0.25, edgecolor="none",
        label=r"Small-Signal Regime  ($P_\mathrm{in} \ll P_s$)")
    saturation_patch = mpatches.Patch(
        facecolor=CORAL, alpha=0.25, edgecolor="none",
        label=r"Saturation Regime  ($P_\mathrm{in} \gg P_s$)")

    # ── Saturation threshold: P_in = P_s ──────────────────────────────────────
    ax.axvline(x=1.0, color=GOLD, linewidth=1.6, linestyle="--")
    ax.text(1.08, 1.3, r"$P_\mathrm{in} = P_s$",
            color=GOLD, fontsize=11, va="bottom", ha="left")

    # ── Unity floor — the hard limit G → 1 ───────────────────────────────────
    ax.axhline(y=1.0, color=CORAL, linewidth=1.2, linestyle=":")
    ax.text(0.011, 1.25, r"$G = 1$  (unity)",
            color=CORAL, fontsize=10, va="bottom", ha="left")

    # ── G_0 annotation at the small-signal plateau ────────────────────────────
    ax.text(0.011, G0 * 0.97, r"$G_0$",
            color=AXES_CLR, fontsize=11, va="top", ha="left")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(0.0, G0 * 1.20)
    ax.set_xlabel(r"Normalised Input Power $P_\mathrm{in}\,/\,P_s$", fontsize=13)
    ax.set_ylabel(r"Power Gain $G$  (dimensionless)", fontsize=13)
    ax.set_title("Gain Compression in the Travelling-Wave Amplifier", fontsize=15, pad=12)
    ax.grid(True)

    # ── Legend (curve + two shaded regions) ───────────────────────────────────
    handles, labels_list = ax.get_legend_handles_labels()
    handles += [small_signal_patch, saturation_patch]
    ax.legend(handles=handles, loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "gain_compression.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
# ── Plot: Gain Coefficient Profile (Lorentzian) ──────────────────────────────
def plot_gain_coefficient_profile():
    nu0 = 50.0         # Arbitrary centre frequency
    delta_nu = 10.0    # Arbitrary FWHM
    gamma_max = 1.0    # Peak gain
    
    nu = np.linspace(nu0 - 30, nu0 + 30, 800)
    gamma = gamma_max / (1 + (2*(nu - nu0)/delta_nu)**2)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # ── Main Lorentzian curve ────────────────────────────────────────────────
    ax.plot(nu, gamma, color=TEAL, linewidth=2.5, label=r"Lorentzian Lineshape $\gamma(\nu)$")
    
    # ── Peak Annotation ──────────────────────────────────────────────────────
    ax.plot(nu0, gamma_max, 'o', color=TEAL, markersize=6)
    # Explaining the bundled peak fully above the curve like the reference image
    ax.text(nu0, gamma_max + 0.04, r"$\gamma_{\max} = N\frac{\lambda^2}{4\pi^2\tau_{sp}\Delta\nu}$", 
            color=TEAL, fontsize=12, va="bottom", ha="center")
    
    # ── FWHM Annotation (Delta nu) ───────────────────────────────────────────
    half_max = gamma_max / 2.0
    nu_left = nu0 - delta_nu / 2.0
    nu_right = nu0 + delta_nu / 2.0
            
    # The span arrow
    ax.annotate("", xy=(nu_left, half_max), xytext=(nu_right, half_max),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=2.0))
    ax.text(nu0, half_max + 0.03, r"$\Delta\nu$", 
            color=CORAL, fontsize=13, ha="center", va="bottom")
            
    # Dot markers at the half-power points
    ax.plot([nu_left, nu_right], [half_max, half_max], 'o', color=CORAL, markersize=5)
    
    # ── Axes Formatting ──────────────────────────────────────────────────────
    ax.set_xlim(nu0 - 25, nu0 + 25)
    ax.set_ylim(0, gamma_max * 1.25)
    
    # X-axis ticks
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=14)
    
    # Harnessing the native grid for perfect full-width dashed lines
    ax.set_yticks([0, half_max, gamma_max])
    ax.set_yticklabels(["0", r"$\frac{1}{2}\gamma_{\max}$", r"$\gamma_{\max}$"], fontsize=13)
    
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Gain Coefficient $\gamma(\nu)$", fontsize=13)
    ax.set_title("Homogeneous Broadening: Lorentzian Gain Profile", fontsize=15, pad=12)
    ax.grid(True)
    
    # ── Legend ───────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    span_line = mlines.Line2D([], [], color=CORAL, linewidth=2.0, label=r"FWHM Bandwidth $\Delta\nu$")
    handles, labels_list = ax.get_legend_handles_labels()
    handles.append(span_line)
    
    ax.legend(handles=handles, loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)
              
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "gain_coefficient_profile.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Plot: Gain Coefficient Profile (Gaussian) ────────────────────────────────
def plot_gain_gaussian_profile():
    nu0 = 50.0         # Arbitrary centre frequency
    sigma_D = 4.25     # Standard deviation parameter
    gamma_max = 1.0    # Peak gain
    
    # Doppler FWHM is explicitly sqrt(8 ln 2) * sigma_D
    delta_nu_D = np.sqrt(8 * np.log(2)) * sigma_D
    
    nu = np.linspace(nu0 - 30, nu0 + 30, 800)
    gamma = gamma_max * np.exp(-((nu - nu0)**2) / (2 * sigma_D**2))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # ── Main Gaussian curve ──────────────────────────────────────────────────
    ax.plot(nu, gamma, color=TEAL, linewidth=2.5, label=r"Gaussian Lineshape $\gamma(\nu)$")
    
    # ── Peak Annotation ──────────────────────────────────────────────────────
    ax.plot(nu0, gamma_max, 'o', color=TEAL, markersize=6)
    # The analytical peak factor bundled together
    ax.text(nu0, gamma_max + 0.04, r"$\gamma_{\max} = N \frac{\lambda^2}{8\pi\tau_{sp}\sigma_D\sqrt{2\pi}}$", 
            color=TEAL, fontsize=12, va="bottom", ha="center")
    
    # ── FWHM Annotation (Delta nu_D) ─────────────────────────────────────────
    half_max = gamma_max / 2.0
    nu_left = nu0 - delta_nu_D / 2.0
    nu_right = nu0 + delta_nu_D / 2.0
    
    ax.annotate("", xy=(nu_left, half_max), xytext=(nu_right, half_max),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=2.0))
    ax.text(nu0, half_max + 0.03, r"$\Delta\nu_D$", 
            color=CORAL, fontsize=13, ha="center", va="bottom")
            
    ax.plot([nu_left, nu_right], [half_max, half_max], 'o', color=CORAL, markersize=5)
    
    # ── Axes Formatting ──────────────────────────────────────────────────────
    ax.set_xlim(nu0 - 25, nu0 + 25)
    ax.set_ylim(0, gamma_max * 1.25)
    
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
    
    ax.set_yticks([0, half_max, gamma_max])
    ax.set_yticklabels(["0", r"$\frac{1}{2}\gamma_{\max}$", r"$\gamma_{\max}$"], fontsize=13)
    
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Gain Coefficient $\gamma(\nu)$", fontsize=13)
    ax.set_title("Inhomogeneous Broadening: Gaussian Gain Profile", fontsize=15, pad=12)
    ax.grid(True)
    
    # ── Legend ───────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    span_line = mlines.Line2D([], [], color=CORAL, linewidth=2.0, label=r"Doppler FWHM $\Delta\nu_D$")
    handles, labels_list = ax.get_legend_handles_labels()
    handles.append(span_line)
    
    ax.legend(handles=handles, loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)
              
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "gain_gaussian_profile.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Plot: Power Gain Profile (G_0) ───────────────────────────────────────────
def plot_gain_power_profile():
    nu0 = 50.0
    delta_nu = 10.0
    G0_max = 10.0
    gammaL_max = np.log(G0_max)
    
    nu = np.linspace(nu0 - 30, nu0 + 30, 800)
    gammaL = gammaL_max / (1 + (2*(nu - nu0)/delta_nu)**2)
    G0 = np.exp(gammaL)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # ── Main G0 curve ────────────────────────────────────────────────────────
    ax.plot(nu, G0, color=TEAL, linewidth=2.5, label=r"Power Gain Profile $G_0(\nu)$")
    
    # ── Peak Annotation ──────────────────────────────────────────────────────
    ax.plot(nu0, G0_max, 'o', color=TEAL, markersize=6)
    ax.text(nu0, G0_max + 0.3, r"$G_{0,\max} = e^{\gamma_{0,\max}L}$", 
            color=TEAL, fontsize=13, va="bottom", ha="center")
    
    # ── FWHM Annotation (Delta nu_G0) ────────────────────────────────────────
    half_max = G0_max / 2.0
    
    # Analytically exact width
    delta_nu_G0 = delta_nu * np.sqrt( np.log(G0_max) / np.log(G0_max / 2.0) - 1 )
    nu_left = nu0 - delta_nu_G0 / 2.0
    nu_right = nu0 + delta_nu_G0 / 2.0
            
    # Span arrow
    ax.annotate("", xy=(nu_left, half_max), xytext=(nu_right, half_max),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=2.0))
    ax.text(nu0, half_max + 0.3, r"$\Delta\nu_{G_0}$", 
            color=CORAL, fontsize=14, ha="center", va="bottom")
            
    # Dot markers at half-power
    ax.plot([nu_left, nu_right], [half_max, half_max], 'o', color=CORAL, markersize=5)
    
    # ── Axes Formatting ──────────────────────────────────────────────────────
    ax.set_xlim(nu0 - 25, nu0 + 25)
    ax.set_ylim(0, G0_max * 1.2)
    
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
    
    # The grid will draw the dotted line for 1/2 G0_max automatically
    ax.set_yticks([1.0, half_max, G0_max])
    ax.set_yticklabels(["1", r"$\frac{1}{2}G_{0,\max}$", r"$G_{0,\max}$"], fontsize=13)
    
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Small-Signal Power Gain $G_0(\nu)$", fontsize=13)
    ax.set_title("Amplifier Bandwidth: Power Gain Profile", fontsize=15, pad=12)
    ax.grid(True)
    
    # ── Legend ───────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    span_line_G0 = mlines.Line2D([], [], color=CORAL, linewidth=2.0, label=r"Amplifier FWHM $\Delta\nu_{G_0}$")
    handles, labels_list = ax.get_legend_handles_labels()
    handles.append(span_line_G0)
    
    ax.legend(handles=handles, loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)
              
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "gain_power_profile.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Plot: Bandwidth Narrowing Cases Overlay ──────────────────────────────────
def plot_bandwidth_narrowing_cases():
    nu0 = 50.0
    delta_nu = 10.0
    nu = np.linspace(nu0 - 30, nu0 + 30, 800)
    
    gamma_profile = 1.0 / (1 + (2*(nu - nu0)/delta_nu)**2)
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    nu_left = nu0 - delta_nu/2.0
    nu_right = nu0 + delta_nu/2.0
    
    # Dotted purple lines for limits, no label in legend
    ax.axvline(nu_left, color=LAVENDER, linestyle=":", lw=1.8, zorder=1)
    ax.axvline(nu_right, color=LAVENDER, linestyle=":", lw=1.8, zorder=1)
    
    def plot_curve(G0_max, col, lbl):
        gammaL_max = np.log(G0_max)
        gammaL = gammaL_max * gamma_profile
        G0 = np.exp(gammaL)
        G0_norm = G0 / G0_max
        ax.plot(nu, G0_norm, color=col, linewidth=2.5, label=lbl, zorder=3)
        return G0_norm
        
    G100 = plot_curve(100.0, PINK,    r"Strong Amplifier ($G_{0,\max} = 100$)")
    G4   = plot_curve(4.0,   TEAL,    r"Boundary Case ($G_{0,\max} = 4$)")
    G2   = plot_curve(2.0,   SKYBLUE, r"Weak Amplifier ($G_{0,\max} = 2$)")
    
    # Red dots for intersection of G=4 with atomic limits, no legend
    ax.plot([nu_left, nu_right], [0.5, 0.5], 'o', color=TEAL, markersize=8, zorder=5)
    
    ax.set_title("Bandwidth Narrowing: Amplifier Power Gain Profiles", fontsize=15, pad=12)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Normalised Amplifier Power Gain", fontsize=14)
    
    # Mark atomic limits firmly in the x-axis tick labels
    ax.set_xticks([nu_left, nu0, nu_right])
    ax.set_xticklabels([r"$\nu_0 - \frac{\Delta\nu}{2}$", r"$\nu_0$", r"$\nu_0 + \frac{\Delta\nu}{2}$"], fontsize=14)
    
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(nu0 - 18, nu0 + 18)
    ax.set_ylim(-0.02, 1.1)
    ax.grid(True)
    
    # Legend just for the 3 curves
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc", fontsize=11)
    
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "bandwidth_narrowing_cases.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    plot_gain_compression()
    plot_gain_coefficient_profile()
    plot_gain_gaussian_profile()
    plot_gain_power_profile()
    plot_bandwidth_narrowing_cases()
