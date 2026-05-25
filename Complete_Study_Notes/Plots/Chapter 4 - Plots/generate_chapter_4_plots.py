"""
generate_chapter_4_plots.py
===========================
Generates matplotlib figures for Chapter 4 of the Optoelectronics notes.

Figures produced:
  - mode_selection_at_turnon.jpg        : Gain profile + loss line + FSR comb showing
                                          which modes survive at laser turn-on.
  - homogeneous_global_saturation.jpg   : Entire Lorentzian gain curve suppressed
                                          uniformly as one mode depletes the inversion.
  - ihb_spectral_hole_burning.jpg       : Gaussian IHB gain profile with narrow spectral
                                          holes burned at specific mode frequencies.
  - hb_vs_ihb_saturation_law.jpg        : Side-by-side comparison of 1/(1+x) vs
                                          1/sqrt(1+x) saturation factors vs flux.
  - output_power_vs_transmission.jpg    : Output power P_out(T) showing optimum mirror
                                          transmissivity T_opt = sqrt(g*l) - l.
  - fp_cavity_round_trip.jpg            : Schematic of the FP cavity showing E-field
                                          labels at each stage of one round trip.
  - airy_function_spectrum.jpg          : Intracavity intensity vs frequency (Airy
                                          function) - resonance peaks, FSR, FWHM, finesse.
  - steady_state_inversion_and_flux.jpg : Saturated steady-state inversion and photon flux
                                          as a function of the pump level N0.
  - lamb_dip.jpg                        : Output power vs mode frequency in a Doppler-
                                          broadened medium showing the central Lamb dip.
  - fp_gain_ripple_spectrum.jpg         : FP amplifier gain G(nu) for sub-threshold G0,
                                          showing G_max, G_min ripple and threshold limit.
  - hb_vs_ihb_mode_spectrum.jpg         : Two-panel steady-state output spectrum -
                                          HB single mode vs IHB multi-mode coexistence.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = SCRIPT_DIR   # Images saved next to script for manual review

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


# ── Helper: Lorentzian lineshape ─────────────────────────────────────────────
def lorentzian(nu, nu0, delta_nu, peak=1.0):
    return peak / (1.0 + (2.0 * (nu - nu0) / delta_nu) ** 2)


# ── Helper: Gaussian lineshape ───────────────────────────────────────────────
def gaussian(nu, nu0, sigma, peak=1.0):
    return peak * np.exp(-0.5 * ((nu - nu0) / sigma) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Mode Selection at Turn-On
# Concept: At t=0, the unsaturated Lorentzian gain profile sits above the flat
# loss line alpha_r over a bandwidth B. Discrete FSR modes (comb) within B are
# the candidates for lasing.  Outside B the gain falls below the loss and those
# modes are immediately excluded.
# ─────────────────────────────────────────────────────────────────────────────
def plot_mode_selection_at_turnon():
    nu0      = 0.0
    delta_nu = 20.0       # FWHM of the atomic lineshape
    gamma0   = 1.0        # peak small-signal gain (normalised)
    alpha_r  = 0.42       # loss level — chosen so ~5 modes survive

    # Continuous gain curve
    nu = np.linspace(-40, 40, 2000)
    gain = lorentzian(nu, nu0, delta_nu, peak=gamma0)

    # FSR comb: place discrete modes evenly
    nu_F = 5.0            # Free Spectral Range
    modes = np.arange(-35, 36, nu_F)
    mode_gains = lorentzian(modes, nu0, delta_nu, peak=gamma0)
    lasing = mode_gains > alpha_r   # modes that survive

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Gain curve
    gain_line, = ax.plot(nu, gain, color=TEAL, linewidth=2.5, label=r"Small-signal gain $\gamma_0$", zorder=3)

    # Loss line
    loss_line = ax.axhline(alpha_r, color=CORAL, linewidth=2.0, linestyle="--",
                           label=r"Total cavity loss $\alpha_r$", zorder=2)

    # Shade the bandwidth B where gain > loss
    gain_above = np.where(gain > alpha_r, gain, np.nan)
    ax.fill_between(nu, alpha_r, gain_above, color=TEAL, alpha=0.10, zorder=1)

    # FSR mode comb — surviving modes in TEAL, dead modes in grey
    for m, g, alive in zip(modes, mode_gains, lasing):
        col  = TEAL if alive else AXES_CLR
        alpha_bar = 0.85 if alive else 0.25
        ax.plot([m, m], [0, g], color=col, linewidth=1.8,
                alpha=alpha_bar, zorder=4)
        ax.plot(m, g, 'o', color=col, markersize=6 if alive else 4,
                alpha=alpha_bar, zorder=5)

    # Bandwidth annotation B
    # Find exact zero-crossings of (gain - alpha_r)
    cross = np.where(np.diff(np.sign(gain - alpha_r)))[0]
    nu_lo = nu[cross[0]]
    nu_hi = nu[cross[-1]]
    
    # Place B arrow above the gain peak to avoid collision with mode comb
    y_B_arrow = gamma0 * 1.15
    ax.annotate("", xy=(nu_lo, y_B_arrow), xytext=(nu_hi, y_B_arrow),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.0))
    # Draw vertical dashed lines to show the projection of B down to the curve
    ax.plot([nu_lo, nu_lo], [alpha_r, y_B_arrow], color=GOLD, linestyle=":", lw=1.5, zorder=1)
    ax.plot([nu_hi, nu_hi], [alpha_r, y_B_arrow], color=GOLD, linestyle=":", lw=1.5, zorder=1)
    
    ax.text(nu0, y_B_arrow + 0.03, r"Gain Bandwidth $B$",
            color=GOLD, fontsize=11, ha="center", va="bottom")

    ax.set_xlim(-42, 42)
    ax.set_ylim(0, gamma0 * 1.40)  # Set origin exactly at 0
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
    ax.set_yticks([0, alpha_r, gamma0])
    ax.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0$"], fontsize=11)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Gain / Loss Coefficient", fontsize=13)
    ax.set_title("Available Gain Bandwidth and Allowed Modes", fontsize=15, pad=12)
    ax.grid(True)

    # Legend
    surviving = mlines.Line2D([], [], color=TEAL, linewidth=2.0,
                               label=r"Surviving modes ($\gamma_0 > \alpha_r$)")
    dead      = mlines.Line2D([], [], color=AXES_CLR, linewidth=1.8,
                               alpha=0.35, label=r"Suppressed modes")
    
    ordered_handles = [gain_line, surviving, dead, loss_line]
    ax.legend(handles=ordered_handles,
              loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "available_gain_bandwidth_modes.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Homogeneous Broadening — Global Gain Saturation
# Concept: As the dominant mode (at nu0) builds up, the ENTIRE Lorentzian is
# suppressed by the same factor 1/(1+Phi/Phi_s).  The gain curve sinks as a
# whole until it just touches alpha_r at nu_q ≈ nu0, and all other modes find
# themselves below alpha_r at the same time — they die.
# ─────────────────────────────────────────────────────────────────────────────
def plot_homogeneous_global_saturation():
    nu0      = 0.0
    delta_nu = 20.0
    gamma0   = 1.0
    alpha_r  = 0.35

    nu = np.linspace(-45, 45, 2000)
    g0 = lorentzian(nu, nu0, delta_nu, peak=gamma0)

    # Saturation levels: Phi/Phi_s = 0 (unsaturated), 0.8, 1.85 (locked to loss)
    # locked level: gamma0/(1+x) = alpha_r  => x = gamma0/alpha_r - 1
    x_lock = gamma0 / alpha_r - 1.0
    levels = [0.0, 0.8, x_lock]
    colors = [TEAL, SKYBLUE, CORAL]
    labels = [
        r"Unsaturated  $\Phi/\Phi_s = 0$",
        r"Partially saturated",
        r"Steady state  $\gamma = \alpha_r$ at $\nu_0$",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for x, col, lbl in zip(levels, colors, labels):
        sat_factor = 1.0 / (1.0 + x)
        ax.plot(nu, g0 * sat_factor, color=col, linewidth=2.5, label=lbl, zorder=3)

    # Loss line
    ax.axhline(alpha_r, color=AXES_CLR, linewidth=1.6, linestyle="--", zorder=2,
               label=r"Cavity loss $\alpha_r$")

    ax.plot(nu0, alpha_r, 'o', color=CORAL, markersize=9, zorder=6)

    # Shade the "dead zone" where steady-state gain < alpha_r
    g_ss = g0 / (1.0 + x_lock)
    ax.fill_between(nu, 0, alpha_r,
                    color=CORAL, alpha=0.07, zorder=1, label="Modes die here")

    ax.set_xlim(-42, 42)
    ax.set_ylim(0, gamma0 * 1.05)
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
    ax.set_yticks([0, alpha_r, gamma0])
    ax.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0$"], fontsize=11)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Gain Coefficient $\gamma$", fontsize=13)
    ax.set_title("Homogeneous Broadening: Global Gain Saturation", fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "homogeneous_global_saturation.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Inhomogeneous Broadening — Spectral Hole Burning
# Concept: Broad Gaussian gain profile (Doppler envelope).  Each lasing mode
# burns a narrow spectral hole only in its own local sub-group of atoms.
# The gain at other frequencies is mostly untouched => multi-mode coexistence.
# ─────────────────────────────────────────────────────────────────────────────
def plot_ihb_spectral_hole_burning():
    nu0      = 0.0
    sigma_D  = 10.0       # Gaussian 1/e half-width
    gamma0   = 1.0
    alpha_r  = 0.30
    delta_H  = 2.0        # homogeneous HWHM per sub-group (narrow)

    nu = np.linspace(-50, 50, 3000)
    g_ihb = gaussian(nu, nu0, sigma_D, peak=gamma0)

    # Three lasing modes equally spaced inside the bandwidth.
    mode_freqs = [-12.0, 0.0, 12.0]

    g_burned = g_ihb.copy()
    for nu_q in mode_freqs:
        g_local  = gaussian(nu_q, nu0, sigma_D, peak=gamma0)
        hole_amp = g_local - alpha_r       # exactly depletes to alpha_r at centre
        if hole_amp > 0:
            g_burned -= hole_amp * lorentzian(nu, nu_q, delta_H, peak=1.0)

    g_burned = np.clip(g_burned, 0, None)  # guard against numerical overshoot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel 1: Origin of IHB (Sub-groups) ──
    l_env1, = ax1.plot(nu, g_ihb, color=TEAL, linewidth=2.5, linestyle="--", zorder=4,
             label=r"Unsaturated gain envelope $\gamma_0$")
    l_loss1 = ax1.axhline(alpha_r, color=AXES_CLR, linewidth=1.6, linestyle="--", zorder=2,
                label=r"Cavity loss $\alpha_r$")

    shifts = np.arange(-36.0, 36.1, 3.0)
    added_bg = False
    added_active = False
    
    l_lasing1_leg = None
    l_indep1_leg = None

    for nu_res in shifts:
        weight = gaussian(nu_res, nu0, sigma_D, peak=gamma0)
        L = lorentzian(nu, nu_res, delta_H, peak=weight)
        
        is_mode = any(np.isclose(nu_res, m) for m in mode_freqs)
        
        if is_mode:
            lbl = "Lasing sub-groups" if not added_active else None
            l, = ax1.plot(nu, L, color=CORAL, linewidth=2.0, alpha=0.9, zorder=3, label=lbl)
            if not added_active:
                l_lasing1_leg = l
            ax1.fill_between(nu, 0, L, color=CORAL, alpha=0.2, zorder=2)
            added_active = True
        else:
            lbl = "Independent atom sub-groups" if not added_bg else None
            l, = ax1.plot(nu, L, color=LAVENDER, linewidth=1.2, alpha=0.6, zorder=1, label=lbl)
            if not added_bg:
                l_indep1_leg = l
            added_bg = True

    ax1.set_xlim(-45, 45)
    ax1.set_ylim(0, gamma0 * 1.35)
    ax1.set_xticks(mode_freqs)
    ax1.set_xticklabels([r"$\nu_{q-1}$", r"$\nu_q$", r"$\nu_{q+1}$"], fontsize=12)
    ax1.set_yticks([0, alpha_r, gamma0])
    ax1.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0$"], fontsize=12)
    ax1.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax1.set_ylabel(r"Gain Coefficient $\bar{\gamma}$", fontsize=13)
    ax1.set_title("1. Inhomogeneous Sub-Groups", fontsize=14, pad=10)
    ax1.grid(True, zorder=-1)
    ax1.legend(handles=[l_env1, l_indep1_leg, l_lasing1_leg, l_loss1], loc="upper right", framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc")

    # ── Panel 2: Hole Burning ──
    l_env2, = ax2.plot(nu, g_ihb, color=TEAL, linewidth=2.5, linestyle="--",
            label=r"Unsaturated gain envelope $\gamma_0$", zorder=2)
    l_burn2, = ax2.plot(nu, g_burned, color=CORAL, linewidth=2.5,
            label=r"Hole-burned gain profile", zorder=3)

    l_loss2 = ax2.axhline(alpha_r, color=AXES_CLR, linewidth=1.6, linestyle="--",
               zorder=2, label=r"Cavity loss $\alpha_r$")

    for nu_q in mode_freqs:
        ax2.plot(nu_q, alpha_r, 'o', color=CORAL, markersize=8, zorder=5)

    import matplotlib.patches as mpatches
    ax2.fill_between(nu, g_burned, g_ihb,
                    where=(g_burned < g_ihb), color=LAVENDER, alpha=0.12, zorder=1)
    shade_proxy2 = mpatches.Patch(color=LAVENDER, alpha=0.12, label="Burned spectral holes")

    ax2.set_xlim(-45, 45)
    ax2.set_ylim(0, gamma0 * 1.35)
    
    ax2.set_xticks(mode_freqs)
    ax2.set_xticklabels([r"$\nu_{q-1}$", r"$\nu_q$", r"$\nu_{q+1}$"], fontsize=12)
    
    ax2.set_yticks([])
    
    ax2.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax2.set_title("2. Selective Depletion", fontsize=14, pad=10)
    ax2.grid(True, zorder=-1)
    
    ax2.legend(handles=[l_env2, shade_proxy2, l_burn2, l_loss2], loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.suptitle("Inhomogeneous Broadening and Spectral Hole Burning", fontsize=16, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ihb_spectral_hole_burning.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: HB vs IHB Saturation Law
# Concept: At the same operating flux, the IHB medium is harder to saturate.
# HB:  gamma = gamma0 / (1 + x)
# IHB: gamma = gamma0 / sqrt(1 + x)   where x = Phi/Phi_s
# This directly explains why IHB lasers support higher intracavity power and
# multi-mode oscillation.
# ─────────────────────────────────────────────────────────────────────────────
def plot_hb_vs_ihb_saturation_law():
    x = np.linspace(0, 10, 800)   # x = Phi / Phi_s

    sat_HB  = 1.0 / (1.0 + x)
    sat_IHB = 1.0 / np.sqrt(1.0 + x)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    line1, = ax.plot(x, sat_HB,  color=TEAL,  linewidth=2.5,
                     label=r"HB (Homogeneous): $\;\dfrac{1}{1+\Phi/\Phi_s}$")
    line2, = ax.plot(x, sat_IHB, color=CORAL, linewidth=2.5,
                     label=r"IHB (Inhomogeneous): $\;\dfrac{1}{\sqrt{1+\Phi/\Phi_s}}$")

    # Mark the saturation point x=1 (Phi = Phi_s) for each curve
    y_HB_at1  = 1.0 / (1.0 + 1.0)
    y_IHB_at1 = 1.0 / np.sqrt(1.0 + 1.0)

    ax.plot(1.0, y_HB_at1, 'o',  color=TEAL,  markersize=8, zorder=5)
    ax.plot(1.0, y_IHB_at1, 'o', color=CORAL, markersize=8, zorder=5)
    
    sat_line = ax.axvline(1.0, color=GOLD, linewidth=1.4, linestyle=":", zorder=1, label=r"Saturation Intensity $\Phi = \Phi_s$")

    # Horizontal guide lines connecting points reading to the y-axis
    ax.plot([0, 1.0], [y_HB_at1, y_HB_at1], color=TEAL, linewidth=1.2, linestyle=":", zorder=1)
    ax.plot([0, 1.0], [y_IHB_at1, y_IHB_at1], color=CORAL, linewidth=1.2, linestyle=":", zorder=1)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    
    # Meaningful Ticks Only
    ax.set_xticks([0, 1.0])
    ax.set_xticklabels(["0", r"$\Phi_s$"], fontsize=12)
    
    ax.set_yticks([y_HB_at1, y_IHB_at1, 1.0])
    ax.set_yticklabels([r"$\frac{1}{2}$", r"$\frac{1}{\sqrt{2}}$", "1"], fontsize=12)
    
    ax.set_xlabel(r"Photon Flux $\Phi$", fontsize=13)
    ax.set_ylabel(r"Saturation Factor (Normalised Gain)", fontsize=13)
    ax.set_title("HB vs. IHB: Gain Saturation Laws Compared", fontsize=15, pad=12)
    ax.grid(True, zorder=-1)
    
    ax.legend(handles=[line1, line2, sat_line], loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "hb_vs_ihb_saturation_law.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Output Power vs Mirror Transmissivity
# Concept: P_out = (P_s/2) * T * (g/(l+T) - 1)
# There is an optimal T_opt = sqrt(g*l) - l that maximises the output.
# Below T_opt: photons trapped inside, little exits.
# Above T_opt: loss too high, gain barely exceeds threshold, little power.
# ─────────────────────────────────────────────────────────────────────────────
def plot_output_power_vs_transmission():
    g = 0.30    # round-trip gain coefficient
    l = 0.04    # round-trip internal loss coefficient

    T_opt = np.sqrt(g * l) - l
    P_max_norm = (np.sqrt(g) - np.sqrt(l)) ** 2   # P_max / (P_s/2)

    T = np.linspace(0.001, g - l, 1200)   # T must keep g > l+T
    P_norm = T * (g / (l + T) - 1.0)              # P_out / (P_s/2)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    line1, = ax.plot(T, P_norm, color=TEAL, linewidth=2.5, zorder=3,
                     label=r"Output Power $P_\mathrm{out}$")

    # Mark the optimum
    line2, = ax.plot(T_opt, P_max_norm, 'o', color=CORAL, markersize=8, zorder=6,
                     label=r"$T_\mathrm{opt} = \sqrt{g\ell} - \ell \quad ; \quad P_{\mathrm{max}} = \frac{P_s}{2}\!\left(\sqrt{g} - \sqrt{\ell}\right)^{\!2}$")
    ax.axvline(T_opt, color=CORAL, linewidth=1.4, linestyle=":", zorder=2)
    ax.axhline(P_max_norm, color=CORAL, linewidth=1.4, linestyle=":", zorder=2)

    # Shade "under-coupling" (T < T_opt) and "over-coupling" (T > T_opt)
    import matplotlib.patches as mpatches
    shade1 = ax.axvspan(0, T_opt, color=TEAL, alpha=0.07, zorder=0)
    shade_proxy1 = mpatches.Patch(color=TEAL, alpha=0.07, label="Under-coupling (light trapped)")
    
    shade2 = ax.axvspan(T_opt, g - l, color=CORAL, alpha=0.07, zorder=0)
    shade_proxy2 = mpatches.Patch(color=CORAL, alpha=0.07, label="Over-coupling (threshold too high)")

    ax.set_xlim(0, g - l)
    ax.set_ylim(0, P_max_norm * 1.05)
    
    ax.set_xlabel(r"Mirror Transmissivity $T = 1 - R$", fontsize=13)
    ax.set_ylabel(r"Output Power $P_\mathrm{out}$", fontsize=13)
    
    # Meaningful Ticks Only
    ax.set_xticks([0, T_opt, g - l])
    ax.set_xticklabels(["0", r"$T_\mathrm{opt}$", r"$g-\ell$"], fontsize=12)
    
    ax.set_yticks([P_max_norm])
    ax.set_yticklabels([r"$P_\mathrm{max}$"], fontsize=12)

    ax.set_title("Optimal Mirror Transmissivity for Maximum Output Power", fontsize=15, pad=12)
    ax.grid(True, zorder=-1)
    
    handles = [line1, line2, shade_proxy1, shade_proxy2]
    labels = [h.get_label() for h in handles]

    ax.legend(handles, labels, loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "output_power_vs_transmission.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5b: Mode Alignment Cases at Turn-On
# Concept: Three cases for how the atomic line centre aligns with the FSR mode comb:
# Case 1: Exact resonance (nu0 = nu_q)
# Case 2: Off-resonance (nu0 between modes)
# Case 3: Symmetric/Degenerate (nu0 exactly halfway between modes)
# ─────────────────────────────────────────────────────────────────────────────
def plot_mode_alignment_cases():
    nu0      = 0.0
    delta_nu = 20.0
    gamma0   = 1.0
    alpha_r  = 0.42
    nu_F     = 8.0  # Make FSR wider for visual clarity

    nu = np.linspace(-30, 30, 1000)
    gain = lorentzian(nu, nu0, delta_nu, peak=gamma0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Case 1: exact resonance
    modes1 = np.arange(-24, 25, nu_F)
    # Case 2: offset by 0.3 * nu_F
    modes2 = np.arange(-24, 25, nu_F) + 0.3 * nu_F
    # Case 3: offset by 0.5 * nu_F
    modes3 = np.arange(-24, 25, nu_F) + 0.5 * nu_F

    titles = ["Case 1: Exact Resonance", "Case 2: Off-Resonance", "Case 3: Symmetric (Degenerate)"]
    mode_lists = [modes1, modes2, modes3]

    for ax, modes, title in zip(axes, mode_lists, titles):
        ax.plot(nu, gain, color=TEAL, linewidth=2.5, zorder=3)
        ax.axhline(alpha_r, color=CORAL, linewidth=2.0, linestyle="--", zorder=2)

        mode_gains = lorentzian(modes, nu0, delta_nu, peak=gamma0)
        lasing = mode_gains > alpha_r
        max_g = np.max(mode_gains)
        is_dominant = np.isclose(mode_gains, max_g)

        for m, g, alive, dom in zip(modes, mode_gains, lasing, is_dominant):
            if dom:
                col = "#dc2626"  # Red for dominant mode
                alpha_bar = 1.0
                msize = 8
                lw = 2.5
                z = 6
            else:
                col  = TEAL if alive else AXES_CLR
                alpha_bar = 0.85 if alive else 0.25
                msize = 6 if alive else 4
                lw = 1.8
                z = 5
                
            ax.plot([m, m], [0, g], color=col, linewidth=lw, alpha=alpha_bar, zorder=z-1)
            ax.plot(m, g, 'o', color=col, markersize=msize, alpha=alpha_bar, zorder=z)

        ax.set_xlim(-25, 25)
        ax.set_ylim(0, gamma0 * 1.35)
        ax.set_xticks([nu0])
        ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
        ax.set_yticks([0, alpha_r, gamma0])
        ax.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0^{\,\max}$"], fontsize=11)
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True)
        ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)

    axes[0].set_ylabel(r"Gain / Loss Coefficient", fontsize=13)

    # Add legend to the first subplot to explain the red dot
    import matplotlib.lines as mlines
    dom_handle = mlines.Line2D([], [], color="#dc2626", marker='o', markersize=8, 
                               linestyle='-', linewidth=2.5, label="Dominant mode(s)")
    axes[0].legend(handles=[dom_handle], loc="upper right", fontsize=11, 
                   framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "mode_alignment_cases.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_mode_selection_at_turnon()
    plot_mode_alignment_cases()
    plot_homogeneous_global_saturation()
    plot_ihb_spectral_hole_burning()
    plot_hb_vs_ihb_saturation_law()
    plot_output_power_vs_transmission()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Fabry-Pérot Cavity Round-Trip Schematic
# Concept: Illustrate the step-by-step field labels E1 → E5 from the text
# derivation. Mirror M1 on left, M2 on right, gain medium in centre.
# Arrows show the propagation direction at each stage.
# ─────────────────────────────────────────────────────────────────────────────
def plot_fp_cavity_round_trip():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ── Cavity walls (mirrors) ────────────────────────────────────────────────
    m1_x, m2_x = 1.2, 8.8
    mirror_h = 3.6
    mirror_y0 = 0.6

    for mx, lbl in [(m1_x, r"$M_1$  ($R_1$)"), (m2_x, r"$M_2$  ($R_2$)")]:
        ax.plot([mx, mx], [mirror_y0, mirror_y0 + mirror_h],
                color=AXES_CLR, linewidth=5, solid_capstyle="butt", zorder=4)
        ax.text(mx, mirror_y0 + mirror_h + 0.15, lbl,
                ha="center", va="bottom", fontsize=11, color=AXES_CLR)

    # ── Gain medium rectangle ─────────────────────────────────────────────────
    gm_x0, gm_x1 = m1_x + 0.05, m2_x - 0.05
    gm_y0, gm_y1 = 1.2, 3.6
    rect = mpatches.FancyBboxPatch((gm_x0, gm_y0), gm_x1 - gm_x0, gm_y1 - gm_y0,
                                   boxstyle="round,pad=0.05",
                                   facecolor=TEAL, alpha=0.12,
                                   edgecolor=TEAL, linewidth=1.8, zorder=2)
    ax.add_patch(rect)
    # Label placed above the box, centred horizontally
    ax.text((gm_x0 + gm_x1) / 2, gm_y1 + 0.1,
            r"Gain Medium $\;(\gamma,\;\alpha_s)$",
            ha="center", va="bottom", fontsize=11, color=TEAL)

    # ── Path parameters ───────────────────────────────────────────────────────
    y1, y2, y3 = 3.2, 2.4, 1.6
    R = 0.4

    # ── Arrow helper ──────────────────────────────────────────────────────────
    def draw_path(x_start, x_end, y, color, label="", label_above=True):
        ax.plot([x_start, x_end], [y, y], color=color, lw=2.2, zorder=3)
        mid_x = (x_start + x_end) / 2
        direction = ">" if x_end > x_start else "<"
        ax.plot(mid_x, y, marker=direction, color=color, markersize=10, zorder=4)
        if label:
            dy = 0.15 if label_above else -0.18
            ax.text(mid_x, y + dy, label,
                    ha="center", va="bottom" if label_above else "top",
                    fontsize=11, color=color)

    # ── Stage 1: E1 → rightward → E2 (at M2) ────────────────────────────────
    draw_path(m1_x, m2_x, y1, TEAL,
              r"$\times\;e^{-j\beta L}\,e^{(\gamma-\alpha_s)L/2}$", label_above=True)
    
    # Start and end dots for Ray 1
    ax.plot(m1_x, y1, 'o', color=TEAL, markersize=6, zorder=5)
    ax.plot(m2_x, y1, 'o', color=TEAL, markersize=6, zorder=5)
    
    ax.text(m1_x + 0.15, y1 + 0.15, r"$E_1$", ha="left", va="bottom", fontsize=12, color=TEAL)
    ax.text(m2_x - 0.15, y1 + 0.15, r"$E_2$", ha="right", va="bottom", fontsize=12, color=TEAL)

    # ── Stage 2: reflection off M2 → E3 ─────────────────────────────────────
    # Placed outside the mirror M2
    ax.text(m2_x + 0.2, (y1 + y2) / 2, r"$\times\,r_2$", ha="left", va="center", fontsize=12, color=CORAL)

    # ── Stage 3: E3 → leftward → E4 (at M1) ────────────────────────────────
    draw_path(m2_x, m1_x, y2, CORAL,
              r"$\times\;e^{-j\beta L}\,e^{(\gamma-\alpha_s)L/2}$", label_above=True)
              
    # Start and end dots for Ray 2
    ax.plot(m2_x, y2, 'o', color=CORAL, markersize=6, zorder=5)
    ax.plot(m1_x, y2, 'o', color=CORAL, markersize=6, zorder=5)
    
    ax.text(m2_x - 0.15, y2 + 0.15, r"$E_3$", ha="right", va="bottom", fontsize=12, color=CORAL)
    ax.text(m1_x + 0.15, y2 + 0.15, r"$E_4$", ha="left", va="bottom", fontsize=12, color=CORAL)

    # ── Stage 4: reflection off M1 → E5 ─────────────────────────────────────
    # Placed outside the mirror M1
    ax.text(m1_x - 0.2, (y2 + y3) / 2, r"$\times\,r_1$", ha="right", va="center", fontsize=12, color=LAVENDER)

    # ── Stage 5: E5 → rightward stub (line stops ~60%; arrow & label at cavity centre)
    stub_end   = m1_x + (m2_x - m1_x) * 0.60
    cavity_mid = (m1_x + m2_x) / 2
    # Draw the stub line
    ax.plot([m1_x, stub_end], [y3, y3], color=LAVENDER, lw=2.2, zorder=3)
    # Arrowhead pinned at true cavity centre (same x as green & red arrows)
    ax.plot(cavity_mid, y3, marker=">", color=LAVENDER, markersize=10, zorder=4)
    # Formula label pinned at true cavity centre, above the line
    ax.text(cavity_mid, y3 + 0.15,
            r"$\times\;e^{-j\beta L}\,e^{(\gamma-\alpha_s)L/2}$",
            ha="center", va="bottom", fontsize=11, color=LAVENDER)
    # Start dot and E_5 label
    ax.plot(m1_x, y3, 'o', color=LAVENDER, markersize=6, zorder=5)
    ax.text(m1_x + 0.15, y3 + 0.15, r"$E_5$", ha="left", va="bottom", fontsize=12, color=LAVENDER)

    # ── Distance L line ───────────────────────────────────────────────────────
    ax.plot([m1_x, m2_x], [0.35, 0.35], color=AXES_CLR, lw=1.5, ls="--")
    ax.plot(m1_x, 0.35, marker="<", color=AXES_CLR, markersize=8)
    ax.plot(m2_x, 0.35, marker=">", color=AXES_CLR, markersize=8)
    ax.text((m1_x + m2_x) / 2, 0.25, r"$L$", ha="center", va="top", fontsize=12, color=AXES_CLR)

    # ── Self-consistency label ────────────────────────────────────────────────
    ax.text(5.0, 0.75,
            r"Sustained oscillation requires: $E_5 = E_1$",
            ha="center", va="bottom", fontsize=11,
            color=AXES_CLR, style="italic")

    ax.set_title("Fabry-Pérot Cavity: One Complete Round Trip", fontsize=15, pad=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fp_cavity_round_trip.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: Airy Function Spectrum
# Concept: The intracavity intensity vs frequency is an Airy function — a comb
# of sharp Lorentzian-like resonance peaks separated by the FSR. Labels FSR,
# FWHM (delta_nu), and the finesse F = FSR / delta_nu.
# ─────────────────────────────────────────────────────────────────────────────
def plot_airy_function_spectrum():
    # Cavity parameters
    R   = 0.80          # geometric mean mirror reflectivity
    als = 0.00          # internal loss (zero for cleaner illustration)
    # Finesse (analytic)
    F_exact = np.pi * R**0.5 / (1 - R)

    # Frequency axis: plot three FSR periods
    FSR   = 1.0         # normalised
    nu    = np.linspace(-0.5 * FSR, 2.5 * FSR, 5000)
    beta_L = np.pi * nu / FSR   # beta * L = pi * nu / FSR so resonances at integers

    denom = (1 - R) ** 2 + 4 * R * np.sin(beta_L) ** 2
    I     = 1.0 / denom          # Airy function (normalised peak = 1/(1-R)^2)
    I_norm = I / I.max()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(nu / FSR, I_norm, color=TEAL, linewidth=2.5, zorder=3)

    # Shade under peaks
    ax.fill_between(nu / FSR, 0, I_norm, color=TEAL, alpha=0.08, zorder=1)

    # ── FSR span arrow ───────────────────────────────────────────────────────
    y_fsr = 1.05
    ax.annotate("", xy=(1.0, y_fsr), xytext=(0.0, y_fsr),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.0))
    ax.text(0.5, y_fsr + 0.02, r"FSR $= \dfrac{c}{2L}$",
            ha="center", va="bottom", fontsize=11, color=GOLD)

    # ── FWHM of one peak — compute numerically ────────────────────────────────
    # Peak at nu=0; half-power = I_norm.max()/2 on left slope
    half_val = 0.5
    # Left crossing of central peak
    left_region = (nu / FSR >= -0.3) & (nu / FSR <= 0.0)
    idx_l = np.where(left_region & (I_norm >= half_val))[0][0]
    nu_l  = nu[idx_l] / FSR
    # Right crossing of central peak
    right_region = (nu / FSR >= 0.0) & (nu / FSR <= 0.3)
    idx_r = np.where(right_region & (I_norm >= half_val))[0][-1]
    nu_r  = nu[idx_r] / FSR

    fwhm_y = half_val
    ax.annotate("", xy=(nu_l, fwhm_y), xytext=(nu_r, fwhm_y),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=2.0))
    ax.text(0.0, fwhm_y - 0.09, r"$\delta\nu$",
            ha="center", va="top", fontsize=12, color=CORAL)

    # ── Finesse annotation ────────────────────────────────────────────────────
    ax.text(1.75, 0.72,
            rf"$\mathcal{{F}} = \dfrac{{\mathrm{{FSR}}}}{{\delta\nu}} \approx {F_exact:.1f}$",
            ha="center", va="center", fontsize=12, color=LAVENDER,
            bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", boxstyle="round,pad=0.3"))

    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.04, 1.25)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r"$\nu_{q}$", r"$\nu_{q+1}$", r"$\nu_{q+2}$"], fontsize=12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", r"$\frac{1}{2}I_{\max}$", r"$I_{\max}$"], fontsize=11)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Normalised Intracavity Intensity", fontsize=13)
    ax.set_title(f"Airy Function Spectrum of the Fabry-Pérot Cavity  ($R = {R}$)",
                 fontsize=15, pad=12)
    ax.grid(True)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "airy_function_spectrum.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8: Gain Saturation — Inversion Depletion and Steady-State Flux
# Panel 1: N/N0 vs Phi/Phi_s  — the gain saturation hyperbola showing how the
#           actual inversion N is depleted hyperbolically as photon flux builds.
# Panel 2: Phi_ss/Phi_s vs N0/N_th — linear growth of intracavity flux above
#           threshold (the "turn-on" curve).
# ─────────────────────────────────────────────────────────────────────────────
def plot_gain_saturation_curve():
    N0_norm = np.linspace(0, 3.5, 1000)  # x-axis: N0 proportional values

    # N (Before threshold: N = N0, after: N = N_th)
    N_ss = np.where(N0_norm < 1.0, N0_norm, 1.0)
    
    # Phi (Before threshold: 0, after climbs linearly)
    Phi_ss = np.where(N0_norm < 1.0, 0.0, N0_norm - 1.0)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    # Plot Inversion N on left axis
    line1 = ax1.plot(N0_norm, N_ss, color=TEAL, linewidth=2.5, 
                     label=r"Steady-State Inversion $N$", zorder=3)
    
    # Plot Photon Flux Phi on right axis
    line2 = ax2.plot(N0_norm, Phi_ss, color=CORAL, linewidth=2.5, 
                     label=r"Steady-State Photon Flux $\Phi$", zorder=3)

    # Threshold marker
    thresh_line = ax1.axvline(1.0, color=GOLD, linewidth=1.8, linestyle="--", 
                              label=r"Lasing Threshold ($N_0 = N_\mathrm{th}$)", zorder=1)

    # Shading the entire above-threshold region (lasing region) vertically, added to legend
    import matplotlib.patches as mpatches
    shade = ax1.axvspan(1.0, 3.2, color=CORAL, alpha=0.08, zorder=0)
    shade_proxy = mpatches.Patch(color=CORAL, alpha=0.08, label="Lasing Region")

    # Labels and Limits
    ax1.set_xlim(0, 3.2)
    ax1.set_ylim(0, 1.5)
    ax2.set_ylim(0, 2.5)

    ax1.set_xlabel(r"Pump Level (Unsaturated Inversion) $N_0$", fontsize=13)
    ax1.set_ylabel(r"Steady-State Inversion $N$", fontsize=13)
    ax2.set_ylabel(r"Steady-State Photon Flux $\Phi$", fontsize=13)
    
    # Meaningful Ticks Only
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["0", r"$N_\mathrm{th}$"], fontsize=12)
    
    # Remove '0' tick to single out the origin point with x-axis zero
    ax1.set_yticks([1])
    ax1.set_yticklabels([r"$N_\mathrm{th}$"], fontsize=12)
    
    ax2.set_yticks([])
    
    ax1.set_title("Steady-State Inversion and Flux vs. Pump Level", fontsize=15, pad=12)
    ax1.grid(True, zorder=-1)

    # Combined legend
    lines = line1 + line2 + [thresh_line]
    labels = [l.get_label() for l in lines]
    
    # Add the proxy artist for the shaded region to the legend
    lines.append(shade_proxy)
    labels.append(shade_proxy.get_label())
    
    ax1.legend(lines, labels, loc="upper left", framealpha=1,
               facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "steady_state_inversion_and_flux.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────
# Plot 9: The Lamb Dip
# Concept: In a Doppler-broadened medium, a mode at nu_q ≠ nu_0 burns TWO
# holes (one per counter-propagating beam, each resonating a different velocity
# class). As nu_q → nu_0 the two holes merge into one → fewer active atoms →
# output power DROPS creating a central dip in the power-vs-frequency curve.
# ─────────────────────────────────────────────────────────────────────────────
def plot_lamb_dip():
    """
    Physically correct Lamb-dip curve (standing-wave cavity, Doppler medium).

    A mode at detuning Delta = nu_q - nu_0 from line centre interacts with
    TWO velocity classes: v = +(Delta/nu_0)*c and v = -(Delta/nu_0)*c.
    The saturated output power is proportional to the sum of both contributions.

    Standard analytic result (Svelto §7.7 / Yariv):
        P(Delta) ∝ [W(+Delta) + W(-Delta)] / sqrt(1 + I/I_s)

    At Delta = 0 both velocity classes coincide (v=0), so only ONE pool of
    atoms is available instead of two.  The transition is modelled with a
    narrow Gaussian overlap of width delta_H (power-broadened homogeneous width).

        P(Delta) ∝ [W(+D) + W(-D) - overlap(D)*W(-D)] / sqrt(1 + I/I_s)

    P_out is then clipped to the Doppler envelope so it is never unphysical.
    """
    sigma_D = 1.0       # Doppler 1/e half-width (normalised units)
    delta_H = 0.10      # power-broadened homogeneous HWHM (much < sigma_D)
    sat     = 3.0       # I / I_s  — drives a clearly visible dip

    nu_q = np.linspace(-2.6 * sigma_D, 2.6 * sigma_D, 2000)

    def W(delta):
        return np.exp(-0.5 * (delta / sigma_D) ** 2)

    denom   = np.sqrt(1.0 + sat)
    overlap = np.exp(-0.5 * (nu_q / delta_H) ** 2)

    # Two-hole contribution (both velocity classes available)
    P_two_hole = (W(nu_q) + W(-nu_q)) / (2.0 * denom)

    # One-hole correction at Delta -> 0 (both classes collapse to v=0)
    P_out = P_two_hole - overlap * W(-nu_q) / (2.0 * denom)

    # Doppler envelope (saturated, but same saturation for a fair comparison)
    g0_env = W(nu_q) / denom   # single-pass saturated envelope

    # Clip: P_out must never exceed the Doppler envelope
    P_out = np.minimum(P_out, g0_env)

    # Normalise both to the peak of the envelope
    norm = g0_env.max()
    g_plot  = g0_env / norm
    P_plot  = P_out  / norm

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(nu_q / sigma_D, g_plot, color=TEAL, linewidth=2.2, linestyle="--",
            label=r"Saturated Doppler envelope", zorder=2)
    ax.plot(nu_q / sigma_D, P_plot, color=CORAL, linewidth=2.5,
            label=r"Mode output power $P_\mathrm{out}(\nu_q)$", zorder=3)
    ax.fill_between(nu_q / sigma_D, 0, P_plot, color=CORAL, alpha=0.08)

    # Mark the dip at nu_q = 0  (Delta = 0)
    idx_centre = np.argmin(np.abs(nu_q))
    dip_val    = P_plot[idx_centre]
    ax.plot(0.0, dip_val, 'o', color=LAVENDER, markersize=9, zorder=6)
    ax.annotate(r"Lamb Dip at $\nu_0$",
                xy=(0.0, dip_val),
                xytext=(0.65, dip_val + 0.16),
                fontsize=11, color=LAVENDER,
                arrowprops=dict(arrowstyle="->", color=LAVENDER, lw=1.5,
                                connectionstyle="arc3,rad=0.3"))

    # Shade the two-hole regime (|Delta| > delta_H)
    hw = 3 * delta_H / sigma_D
    ax.axvspan(-2.5, -hw, color=SKYBLUE, alpha=0.06)
    ax.axvspan( hw,  2.5, color=SKYBLUE, alpha=0.06)
    ax.text(-1.4, 0.08, "Two holes\nburned", color=SKYBLUE, fontsize=9, ha="center")
    ax.text( 1.4, 0.08, "Two holes\nburned", color=SKYBLUE, fontsize=9, ha="center")

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.04, 1.18)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels([r"$-2\sigma_D$", r"$-\sigma_D$", r"$\nu_0$",
                        r"$+\sigma_D$", r"$+2\sigma_D$"], fontsize=11)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(r"Mode Frequency $\nu_q$", fontsize=13)
    ax.set_ylabel(r"Normalised Output Power", fontsize=13)
    ax.set_title("The Lamb Dip: Spectral Hole Burning in a Standing-Wave Cavity",
                 fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lamb_dip.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 10: FP Amplifier Gain Ripple Spectrum
# Exact formula: G = (1-R1)(1-R2)*G0 / [(1-G0*sqrt(R1R2))^2 + 4*G0*sqrt(R1R2)*sin²(βL)]
# Show for two G0 values: moderate and near-threshold. Annotate G_max, G_min, ripple.
# ─────────────────────────────────────────────────────────────────────────────
def plot_fp_gain_ripple_spectrum():
    """
    FP amplifier gain G(nu) for two sub-threshold G0 values.
    Y-axis clipped to 5x G_max(moderate) so BOTH curves are visible:
    - Moderate case: both G_max and G_min are fully visible with ripple annotation.
    - Near-threshold: peaks clip off the top (shown by upward arrows), making
      the point that they diverge as G0*sqrt(R1R2) -> 1.
    """
    R1, R2 = 0.70, 0.70
    sqrtR  = np.sqrt(R1 * R2)

    phi     = np.linspace(0, 3 * np.pi, 6000)
    nu_norm = phi / np.pi   # 0 -> 3 (three FSR periods shown)

    def G_fp(G0, phi):
        num = (1 - R1) * (1 - R2) * G0
        den = (1 - G0 * sqrtR) ** 2 + 4.0 * G0 * sqrtR * np.sin(phi) ** 2
        return num / den

    def Gmax(G0): return (1 - R1) * (1 - R2) * G0 / (1 - G0 * sqrtR) ** 2
    def Gmin(G0): return (1 - R1) * (1 - R2) * G0 / (1 + G0 * sqrtR) ** 2

    G0_mod  = 0.80              # moderate — clearly sub-threshold
    G0_near = 1.0 / sqrtR * 0.93   # 93 % of threshold

    G_mod  = G_fp(G0_mod,  phi)
    G_near = G_fp(G0_near, phi)

    gmax_m = Gmax(G0_mod)
    gmin_m = Gmin(G0_mod)
    gmax_n = Gmax(G0_near)

    # Clip y-axis: 5x moderate G_max makes the moderate curve fully visible
    y_clip = 5.0 * gmax_m

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.plot(nu_norm, np.clip(G_mod,  0, y_clip * 1.05),
            color=TEAL,  linewidth=2.5,
            label=rf"$G_0 = {G0_mod}$ (moderate sub-threshold)")
    ax.plot(nu_norm, np.clip(G_near, 0, y_clip * 1.05),
            color=CORAL, linewidth=2.5,
            label=rf"$G_0 = {G0_near:.2f}$ (near threshold, $93\%$)")

    # Moderate case: dotted guide lines + ripple annotation
    ax.axhline(gmax_m, color=TEAL, linewidth=1.0, linestyle=":")
    ax.axhline(gmin_m, color=TEAL, linewidth=1.0, linestyle=":")

    ripple_m = gmax_m / gmin_m
    ax.annotate("", xy=(2.72, gmax_m), xytext=(2.72, gmin_m),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.8))
    ax.text(2.75, (gmax_m + gmin_m) / 2,
            rf"$\rho = {ripple_m:.1f}$",
            color=TEAL, fontsize=10, va="center")

    ax.text(2.97, gmax_m + 0.01, r"$G_\mathrm{max}$",
            color=TEAL, fontsize=9, ha="right", va="bottom")
    ax.text(2.97, gmin_m - 0.01, r"$G_\mathrm{min}$",
            color=TEAL, fontsize=9, ha="right", va="top")

    # Near-threshold: mark clipped peaks with upward arrows
    for q in [0, 1, 2]:
        nu_peak = 2 * q
        if nu_peak < nu_norm[-1]:
            ax.annotate("", xy=(nu_peak, y_clip * 1.00),
                        xytext=(nu_peak, y_clip * 0.86),
                        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.8))
    ax.text(1.0, y_clip * 0.88,
            rf"$G_{{\mathrm{{max}}}}^{{(\mathrm{{near}})}} \approx {gmax_n:.0f}$",
            color=CORAL, fontsize=9, ha="center", va="top")

    ax.set_xlim(0, 3)
    ax.set_ylim(-0.05, y_clip)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        [r"$\nu_q$",
         r"$\nu_q + \frac{\mathrm{FSR}}{2}$",
         r"$\nu_{q+1}$",
         r"$\nu_{q+1}+\frac{\mathrm{FSR}}{2}$"],
        fontsize=10)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"FP Amplifier Power Gain $G(\nu)$", fontsize=13)
    ax.set_title("Fabry-Pérot Amplifier: Gain Ripple Spectrum", fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper center", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fp_gain_ripple_spectrum.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 11: HB vs IHB — Steady-State Mode Spectrum
# Two panels, shared x-axis.
# Left: HB laser — gain envelope + alpha_r line; one surviving mode spike.
# Right: IHB laser — Gaussian envelope + alpha_r line; multiple mode spikes.
# ─────────────────────────────────────────────────────────────────────────────
def plot_hb_vs_ihb_mode_spectrum():
    nu0      = 0.0
    nu_F     = 4.0       # FSR
    alpha_r  = 0.38

    # HB: Lorentzian gain, globally saturated to alpha_r at nu0
    delta_H  = 20.0
    gamma0_H = 1.0
    nu_HB    = np.linspace(-45, 45, 2000)
    g_HB_unsat = gamma0_H / (1 + (2 * (nu_HB - nu0) / delta_H) ** 2)

    # Saturation factor: whole curve sinks until peak touches alpha_r
    sat_H = alpha_r / gamma0_H
    g_HB_sat = g_HB_unsat * sat_H

    # IHB: Gaussian gain (unsaturated envelope remains largely intact)
    sigma_I  = 9.0
    gamma0_I = 1.0
    nu_IHB   = np.linspace(-45, 45, 2000)
    g_IHB    = gamma0_I * np.exp(-0.5 * (nu_IHB / sigma_I) ** 2)

    # Discrete modes
    modes = np.arange(-40, 41, nu_F)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

    # ── Left panel: Homogeneous Broadening ───────────────────────────────────
    l_unsat1, = ax1.plot(nu_HB, g_HB_unsat, color=TEAL, linewidth=2.2, linestyle="--",
             alpha=0.6, label=r"Unsaturated gain envelope $\gamma_0$")
    l_sat1, = ax1.plot(nu_HB, g_HB_sat,   color=TEAL, linewidth=2.5,
             label=r"Globally saturated gain $\gamma$")
    l_loss1 = ax1.axhline(alpha_r, color=AXES_CLR, linewidth=1.6, linestyle="--",
                label=r"Cavity loss $\alpha_r$")

    l_mode_alive1 = None
    l_mode_dead1  = None

    # Only the mode at nu0 survives
    for m in modes:
        g_local = gamma0_H * sat_H / (1 + (2 * (m - nu0) / delta_H) ** 2)
        alive = abs(m - nu0) < nu_F / 2
        col  = CORAL if alive else AXES_CLR
        alph = 0.9 if alive else 0.4
        lw   = 2.5 if alive else 1.5
        ms   = 8 if alive else 4
        
        l_m, = ax1.plot([m, m], [0, g_local], color=col, linewidth=lw, alpha=alph)
        ax1.plot(m, g_local, 'o', color=col, markersize=ms, alpha=alph)
        
        if alive and not l_mode_alive1:
            l_m.set_label("Lasing mode")
            l_mode_alive1 = l_m
        elif not alive and not l_mode_dead1:
            l_m.set_label("Suppressed modes")
            l_mode_dead1 = l_m

    ax1.set_title("Homogeneous Broadening\n(Global Saturation → Single Mode)",
                  fontsize=13, pad=8)
    ax1.set_xlim(-42, 42)
    ax1.set_ylim(0, 1.65)
    ax1.set_xticks([nu0])
    ax1.set_xticklabels([r"$\nu_0$"], fontsize=12)
    ax1.set_yticks([0, alpha_r, gamma0_H])
    ax1.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0$"], fontsize=12)
    ax1.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax1.set_ylabel(r"Gain / Mode Power", fontsize=13)
    ax1.grid(True, zorder=-1)
    
    handles1 = [l_unsat1, l_sat1]
    if l_mode_alive1: handles1.append(l_mode_alive1)
    if l_mode_dead1: handles1.append(l_mode_dead1)
    handles1.append(l_loss1)
    
    ax1.legend(handles=handles1, fontsize=10, loc="upper right",
               framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc")

    # ── Right panel: Inhomogeneous Broadening ─────────────────────────────────
    l_unsat2, = ax2.plot(nu_IHB, g_IHB, color=TEAL, linewidth=2.5, linestyle="--",
             label=r"Unsaturated gain envelope $\bar{\gamma}_0$")
    l_loss2 = ax2.axhline(alpha_r, color=AXES_CLR, linewidth=1.6, linestyle="--",
                label=r"Cavity loss $\alpha_r$")

    l_mode_alive2 = None
    l_mode_dead2  = None

    # All modes within bandwidth B survive independently
    for m in modes:
        g_local = gamma0_I * np.exp(-0.5 * (m / sigma_I) ** 2)
        alive = g_local > alpha_r
        col  = CORAL if alive else AXES_CLR
        alph = 0.9 if alive else 0.4
        lw   = 2.5 if alive else 1.5
        ms   = 8 if alive else 4
        
        l_m, = ax2.plot([m, m], [0, g_local], color=col, linewidth=lw, alpha=alph)
        ax2.plot(m, g_local, 'o', color=col, markersize=ms, alpha=alph)
        
        if alive and not l_mode_alive2:
            l_m.set_label("Lasing modes")
            l_mode_alive2 = l_m
        elif not alive and not l_mode_dead2:
            l_m.set_label("Suppressed modes")
            l_mode_dead2 = l_m

    # B annotation: vertical dashed lines from alpha_r intersection up above curve,
    # then a solid double-tipped arrow between them with label B
    nu_lo_I = -sigma_I * np.sqrt(-2 * np.log(alpha_r / gamma0_I))
    nu_hi_I =  sigma_I * np.sqrt(-2 * np.log(alpha_r / gamma0_I))
    B_arrow_y = 1.06  # just above the curve peak

    # Vertical golden dashed lines from intersection up above the curve
    ax2.plot([nu_lo_I, nu_lo_I], [alpha_r, B_arrow_y], color=GOLD,
             linewidth=1.5, linestyle="--", zorder=2)
    ax2.plot([nu_hi_I, nu_hi_I], [alpha_r, B_arrow_y], color=GOLD,
             linewidth=1.5, linestyle="--", zorder=2)

    # Solid double-tipped arrow at the top between the two vertical lines
    ax2.annotate("", xy=(nu_hi_I, B_arrow_y),
                 xytext=(nu_lo_I, B_arrow_y),
                 arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.0))
    ax2.text((nu_lo_I + nu_hi_I) / 2, B_arrow_y + 0.03, r"$B$",
             ha="center", va="bottom", fontsize=13, color=GOLD, fontweight="bold")

    ax2.set_title("Inhomogeneous Broadening\n(Spectral Hole Burning → Multi-Mode)",
                  fontsize=13, pad=8)
    ax2.set_xlim(-42, 42)
    ax2.set_ylim(0, 1.65)
    ax2.set_xticks([nu0])
    ax2.set_xticklabels([r"$\nu_0$"], fontsize=12)
    ax2.set_yticks([])
    ax2.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax2.grid(True, zorder=-1)
    
    handles2 = [l_unsat2]
    if l_mode_alive2: handles2.append(l_mode_alive2)
    if l_mode_dead2: handles2.append(l_mode_dead2)
    handles2.append(l_loss2)
    
    ax2.legend(handles=handles2, fontsize=10, loc="upper right",
               framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.suptitle("Steady-State Mode Spectrum: Homogeneously vs. Inhomogeneously Broadened Laser",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "hb_vs_ihb_mode_spectrum.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 12: Frequency Pulling Diagram
# Concept: Two-panel figure.
# Top: Lorentzian gain profile + cold resonator mode comb (dashed) and hot
#      resonator comb (solid lavender), both overlaid with the loss line.
# Bottom: Pulling arrows showing the direction and magnitude of each mode's
#         shift from nu_q toward nu_0 via the formula nu'_q=(nu_q+P*nu0)/(1+P).
# ─────────────────────────────────────────────────────────────────────────────
def plot_frequency_pulling_diagram():
    nu0      = 0.0
    delta_nu = 30.0
    gamma0   = 1.0
    alpha_r  = 0.25
    nu_F     = 8.0
    P        = 0.35   # dimensionless pulling parameter c*gamma/(2*pi*Delta_nu)

    nu   = np.linspace(-50, 50, 3000)
    gain = lorentzian(nu, nu0, delta_nu, peak=gamma0)

    cold_modes = np.arange(-40, 41, nu_F)
    hot_modes  = (cold_modes + P * nu0) / (1.0 + P)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [1.6, 1]},
        sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # -- Top panel --
    ax_top.plot(nu, gain, color=TEAL, linewidth=2.5, zorder=3,
                label=r"Gain profile $\gamma_0$")
    ax_top.axhline(alpha_r, color=CORAL, linewidth=1.8, linestyle="--", zorder=2,
                   label=r"Cavity loss $\alpha_r$")
    ax_top.axvline(nu0, color=GOLD, linewidth=1.2, linestyle=":", zorder=1)
    ax_top.text(nu0 + 0.8, gamma0 * 1.05, r"$\nu_0$", color=GOLD, fontsize=12)

    for m in cold_modes:
        g = lorentzian(m, nu0, delta_nu, peak=gamma0)
        ax_top.plot([m, m], [0, g], color=AXES_CLR, linewidth=1.4,
                    linestyle="--", alpha=0.5, zorder=2)

    for m_h, m_c in zip(hot_modes, cold_modes):
        g = lorentzian(m_c, nu0, delta_nu, peak=gamma0)
        ax_top.plot([m_h, m_h], [0, g], color=LAVENDER, linewidth=2.0, zorder=3)

    import matplotlib.lines as mlines
    cold_line = mlines.Line2D([], [], color=AXES_CLR, linewidth=1.4,
                               linestyle="--", alpha=0.6, label=r"Cold modes $\nu_q$")
    hot_line  = mlines.Line2D([], [], color=LAVENDER, linewidth=2.0,
                               label=r"Hot modes $\nu'_q$")
    handles, _ = ax_top.get_legend_handles_labels()
    ax_top.legend(handles=handles + [cold_line, hot_line],
                  loc="upper right", framealpha=1,
                  facecolor="#f5f5f5", edgecolor="#cccccc", fontsize=10)

    ax_top.set_ylim(-0.04, gamma0 * 1.30)
    ax_top.set_yticks([0, alpha_r, gamma0])
    ax_top.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0$"], fontsize=11)
    ax_top.set_ylabel(r"Gain / Loss", fontsize=13)
    ax_top.set_title("Frequency Pulling: Cold vs Hot Resonator Modes",
                     fontsize=15, pad=12)
    ax_top.grid(True)

    # -- Bottom panel: pulling arrows --
    ax_bot.axhline(0, color=AXES_CLR, linewidth=1.0, linestyle=":")
    ax_bot.axvline(nu0, color=GOLD, linewidth=1.2, linestyle=":", zorder=1)

    for m_c, m_h in zip(cold_modes, hot_modes):
        if abs(m_h - m_c) > 0.05:
            ax_bot.annotate("", xy=(m_h, 0), xytext=(m_c, 0),
                            arrowprops=dict(arrowstyle="->", color=LAVENDER,
                                            lw=1.8, mutation_scale=12))
        ax_bot.plot(m_c, 0, 'o', color=AXES_CLR, markersize=5, alpha=0.5, zorder=4)
        ax_bot.plot(m_h, 0, 'o', color=LAVENDER, markersize=6, zorder=5)

    ax_bot.set_xlim(-44, 44)
    ax_bot.set_ylim(-0.5, 0.5)
    ax_bot.set_yticks([])
    ax_bot.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax_bot.set_ylabel(r"Pulling", fontsize=12)
    ax_bot.text(nu0 + 1, 0.35, r"Modes pulled toward $\nu_0$",
                color=LAVENDER, fontsize=10, ha="left")
    ax_bot.grid(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "frequency_pulling_diagram.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 13: Frequency Chirping — Direct Laser Modulation
# Concept: Two stacked panels sharing the time axis.
# Top:    Output power follows a 1-0-1-1 square-wave bit pattern.
# Bottom: Instantaneous frequency deviation delta_nu(t) — NOT constant.
#         It swings positive on rising edges and negative on falling edges,
#         following ~ d(ln P)/dt (transient chirp) plus a small adiabatic term.
# ─────────────────────────────────────────────────────────────────────────────
def plot_frequency_chirping():
    t = np.linspace(0, 4, 4000)
    P_high, P_low, rise = 1.0, 0.25, 0.05

    def smooth_pulse(t, t0, t1):
        return (P_low + (P_high - P_low) *
                (0.5 * np.tanh((t - t0) / rise) - 0.5 * np.tanh((t - t1) / rise)))

    # Bit pattern: 1, 0, 1, 1
    P_out = smooth_pulse(t, 0.0, 1.0) + smooth_pulse(t, 2.0, 4.0)
    P_out = np.clip(P_out, P_low, P_high)

    # Transient chirp ~ d(ln P)/dt; adiabatic chirp ~ P
    dln_P_dt = np.gradient(np.log(P_out + 1e-6), t)
    adiabatic = 0.03 * (P_out - P_low) / (P_high - P_low)
    nu_inst = 0.22 * dln_P_dt + adiabatic

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6),
                                    sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    # -- Top: power --
    ax1.plot(t, P_out, color=TEAL, linewidth=2.5, zorder=3)
    ax1.fill_between(t, P_low, P_out, color=TEAL, alpha=0.10)
    ax1.axhline(P_high, color=AXES_CLR, linewidth=1.0, linestyle=":", alpha=0.5)
    ax1.axhline(P_low,  color=AXES_CLR, linewidth=1.0, linestyle=":", alpha=0.5)
    ax1.text(0.04, P_high + 0.05, r"$P_\mathrm{high}$  (Logic 1)",
             color=TEAL, fontsize=10)
    ax1.text(0.04, P_low  - 0.08, r"$P_\mathrm{low}$  (Logic 0)",
             color=TEAL, fontsize=10)
    ax1.set_ylabel(r"Output Power", fontsize=13)
    ax1.set_ylim(-0.1, 1.35)
    ax1.set_yticks([P_low, P_high])
    ax1.set_yticklabels([r"$P_\mathrm{low}$", r"$P_\mathrm{high}$"], fontsize=10)
    ax1.set_title("Direct Laser Modulation and Frequency Chirping",
                  fontsize=15, pad=12)
    ax1.grid(True)

    # -- Bottom: instantaneous frequency deviation --
    ax2.plot(t, nu_inst, color=CORAL, linewidth=2.5, zorder=3)
    ax2.fill_between(t, 0, nu_inst, where=(nu_inst > 0), color=CORAL,   alpha=0.10)
    ax2.fill_between(t, 0, nu_inst, where=(nu_inst < 0), color=LAVENDER, alpha=0.10)
    ax2.axhline(0, color=AXES_CLR, linewidth=1.2, linestyle="--", alpha=0.7)

    ax2.text(0.42,  0.14, "Rising edge:\n" + r"$\nu$ swings up",
             color=CORAL, fontsize=9, ha="center")
    ax2.text(1.08, -0.15, "Falling edge:\n" + r"$\nu$ swings down",
             color=LAVENDER, fontsize=9, ha="center")

    ax2.set_ylabel(r"Frequency Deviation $\delta\nu(t)$", fontsize=13)
    ax2.set_xlabel(r"Time (bit periods)", fontsize=13)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-0.45, 0.45)
    ax2.set_yticks([-0.3, 0, 0.3])
    ax2.set_yticklabels(
        [r"$-\delta\nu_\mathrm{max}$", "0", r"$+\delta\nu_\mathrm{max}$"],
        fontsize=10)
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(
        [r"$0$", r"$T_b$", r"$2T_b$", r"$3T_b$", r"$4T_b$"],
        fontsize=11)
    ax2.grid(True)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "frequency_chirping.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Original 5
    plot_mode_selection_at_turnon()
    plot_homogeneous_global_saturation()
    plot_ihb_spectral_hole_burning()
    plot_hb_vs_ihb_saturation_law()
    plot_output_power_vs_transmission()
    # Physics-derivation figures
    plot_fp_cavity_round_trip()
    plot_airy_function_spectrum()
    plot_gain_saturation_curve()
    plot_lamb_dip()
    plot_fp_gain_ripple_spectrum()
    plot_hb_vs_ihb_mode_spectrum()
    # Hot resonator figures
    plot_frequency_pulling_diagram()
    plot_frequency_chirping()
