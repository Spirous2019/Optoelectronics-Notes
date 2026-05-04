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
  - gain_saturation_curve.jpg           : Saturated vs unsaturated gain as a function of
                                          pump level, locking to alpha_r at steady state.
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
    ax.plot(nu, gain, color=TEAL, linewidth=2.5, label=r"Small-signal gain $\gamma_0(\nu)$", zorder=3)

    # Loss line
    ax.axhline(alpha_r, color=CORAL, linewidth=2.0, linestyle="--",
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
    ax.annotate("", xy=(nu_lo, alpha_r + 0.09), xytext=(nu_hi, alpha_r + 0.09),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.0))
    ax.text(nu0, alpha_r + 0.12, r"Gain Bandwidth $B$",
            color=GOLD, fontsize=11, ha="center", va="bottom")

    # alpha_r label
    ax.text(35.5, alpha_r + 0.025, r"$\alpha_r$",
            color=CORAL, fontsize=12, ha="left", va="bottom")

    ax.set_xlim(-42, 42)
    ax.set_ylim(-0.05, gamma0 * 1.35)
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
    ax.set_yticks([0, alpha_r, gamma0])
    ax.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0^{\,\max}$"], fontsize=11)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Gain / Loss Coefficient", fontsize=13)
    ax.set_title("Mode Selection at Laser Turn-On", fontsize=15, pad=12)
    ax.grid(True)

    # Legend
    surviving = mlines.Line2D([], [], color=TEAL, linewidth=2.0,
                               label=r"Surviving modes ($\gamma_0 > \alpha_r$)")
    dead      = mlines.Line2D([], [], color=AXES_CLR, linewidth=1.8,
                               alpha=0.35, label=r"Suppressed modes")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [surviving, dead],
              loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "mode_selection_at_turnon.jpg")
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
        r"Partially saturated  $\Phi/\Phi_s = 0.8$",
        r"Steady state  $\gamma = \alpha_r$ at $\nu_0$",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for x, col, lbl in zip(levels, colors, labels):
        sat_factor = 1.0 / (1.0 + x)
        ax.plot(nu, g0 * sat_factor, color=col, linewidth=2.5, label=lbl, zorder=3)

    # Loss line
    ax.axhline(alpha_r, color=AXES_CLR, linewidth=1.6, linestyle="--", zorder=2,
               label=r"Cavity loss $\alpha_r$")

    # Mark the single surviving lasing frequency.
    # At nu=nu0: g0(nu0) = gamma0; sat_factor_lock = alpha_r/gamma0
    # => g0(nu0) * sat_factor_lock = alpha_r  — the dot sits exactly on the loss line.
    ax.plot(nu0, alpha_r, 'o', color=CORAL, markersize=9, zorder=6)
    ax.annotate(r"Single lasing mode at $\nu_q \approx \nu_0$",
                xy=(nu0, alpha_r), xytext=(13, alpha_r + 0.22),
                fontsize=10, color=CORAL,
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5,
                                connectionstyle="arc3,rad=-0.25"))

    # Shade the "dead zone" where steady-state gain < alpha_r
    g_ss = g0 / (1.0 + x_lock)
    ax.fill_between(nu, g_ss, alpha_r,
                    where=(g_ss < alpha_r), color=CORAL, alpha=0.07, zorder=1)

    # Label inside the dead zone (below alpha_r)
    ax.text(28, alpha_r * 0.55, r"Modes die here",
            color=CORAL, fontsize=9.5, ha="left", va="center", alpha=0.8)

    ax.set_xlim(-42, 42)
    ax.set_ylim(-0.04, gamma0 * 1.25)
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
    ax.set_yticks([0, alpha_r, gamma0])
    ax.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0^{\,\max}$"], fontsize=11)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Gain Coefficient $\gamma(\nu,\,\Phi)$", fontsize=13)
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
    # Hole depth at each mode: the burned profile floor sits exactly at alpha_r.
    # hole_amp = g_ihb(nu_q) - alpha_r  (depth so minimum = alpha_r)
    mode_freqs = [-12.0, 0.0, 12.0]

    g_burned = g_ihb.copy()
    for nu_q in mode_freqs:
        g_local  = gaussian(nu_q, nu0, sigma_D, peak=gamma0)
        hole_amp = g_local - alpha_r       # exactly depletes to alpha_r at centre
        if hole_amp > 0:
            g_burned -= hole_amp * lorentzian(nu, nu_q, delta_H, peak=1.0)

    g_burned = np.clip(g_burned, 0, None)  # guard against numerical overshoot

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(nu, g_ihb,    color=TEAL,    linewidth=2.5, linestyle="--",
            label=r"Unsaturated gain envelope $\gamma_0(\nu)$", zorder=2)
    ax.plot(nu, g_burned, color=SKYBLUE, linewidth=2.5,
            label=r"Hole-burned gain profile", zorder=3)

    # Loss line
    ax.axhline(alpha_r, color=AXES_CLR, linewidth=1.6, linestyle="--",
               zorder=2, label=r"Cavity loss $\alpha_r$")

    # Mark the exact bottom of each hole (which sits at alpha_r)
    for nu_q in mode_freqs:
        ax.plot(nu_q, alpha_r, 'o', color=CORAL, markersize=7, zorder=5)

    # Shade hole regions
    ax.fill_between(nu, g_burned, g_ihb,
                    where=(g_burned < g_ihb), color=LAVENDER, alpha=0.12, zorder=1)

    # Mode frequency labels: placed just above the alpha_r line at each mode centre
    label_y = alpha_r + 0.04
    ax.text(-12, label_y, r"$\nu_{q-1}$", color=CORAL, fontsize=11, ha="center")
    ax.text(  0, label_y, r"$\nu_q$",     color=CORAL, fontsize=11, ha="center")
    ax.text( 12, label_y, r"$\nu_{q+1}$", color=CORAL, fontsize=11, ha="center")

    # Annotate gain intact between holes
    ax.annotate("Gain intact\nbetween holes",
                xy=(6, gaussian(6, nu0, sigma_D, peak=gamma0) - 0.04),
                xytext=(22, 0.72),
                fontsize=9.5, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4,
                                connectionstyle="arc3,rad=0.2"))

    ax.set_xlim(-45, 45)
    ax.set_ylim(-0.04, gamma0 * 1.25)
    ax.set_xticks([nu0])
    ax.set_xticklabels([r"$\nu_0$"], fontsize=13)
    ax.set_yticks([0, alpha_r, gamma0])
    ax.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0^{\,\max}$"], fontsize=11)
    ax.set_xlabel(r"Optical Frequency $\nu$", fontsize=13)
    ax.set_ylabel(r"Gain Coefficient $\bar{\gamma}(\nu)$", fontsize=13)
    ax.set_title("Inhomogeneous Broadening: Spectral Hole Burning", fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

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

    ax.plot(x, sat_HB,  color=TEAL,  linewidth=2.5,
            label=r"HB (Homogeneous): $\;\dfrac{1}{1+\Phi/\Phi_s}$")
    ax.plot(x, sat_IHB, color=CORAL, linewidth=2.5,
            label=r"IHB (Inhomogeneous): $\;\dfrac{1}{\sqrt{1+\Phi/\Phi_s}}$")

    # Mark the saturation point x=1 (Phi = Phi_s) for each curve
    y_HB_at1  = 1.0 / (1.0 + 1.0)
    y_IHB_at1 = 1.0 / np.sqrt(1.0 + 1.0)

    ax.plot(1.0, y_HB_at1, 'o',  color=TEAL,  markersize=8, zorder=5)
    ax.plot(1.0, y_IHB_at1, 'o', color=CORAL, markersize=8, zorder=5)
    ax.axvline(1.0, color=GOLD, linewidth=1.4, linestyle=":", zorder=1)
    ax.text(1.06, 0.92, r"$\Phi = \Phi_s$",
            color=GOLD, fontsize=11, va="top")

    # Shade the gap between the two curves (IHB advantage region)
    ax.fill_between(x, sat_HB, sat_IHB, color=LAVENDER, alpha=0.10, zorder=0)
    ax.text(5.5, 0.50, r"IHB retains more gain",
            color=LAVENDER, fontsize=10, ha="center", va="center",
            style="italic")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel(r"Normalised Photon Flux $\Phi\,/\,\Phi_s$", fontsize=13)
    ax.set_ylabel(r"Saturation Factor (normalised gain)", fontsize=13)
    ax.set_title("HB vs. IHB: Gain Saturation Laws Compared", fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1,
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

    T = np.linspace(0.001, g - l - 0.001, 1200)   # T must keep g > l+T
    P_norm = T * (g / (l + T) - 1.0)              # P_out / (P_s/2)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(T, P_norm, color=TEAL, linewidth=2.5, zorder=3)

    # Mark the optimum
    ax.plot(T_opt, P_max_norm, 'o', color=CORAL, markersize=10, zorder=6,
            label=r"$T_\mathrm{opt} = \sqrt{g\ell} - \ell$")
    ax.axvline(T_opt, color=CORAL, linewidth=1.4, linestyle=":", zorder=2)
    ax.axhline(P_max_norm, color=CORAL, linewidth=1.4, linestyle=":", zorder=2)

    # Shade "under-coupling" (T < T_opt) and "over-coupling" (T > T_opt)
    ax.axvspan(T[0], T_opt,  color=TEAL,  alpha=0.07, zorder=0)
    ax.axvspan(T_opt, T[-1], color=CORAL, alpha=0.07, zorder=0)

    ax.text(T_opt / 2, P_max_norm * 0.45,
            "Under-coupling\n(light trapped inside)",
            color=TEAL, fontsize=9.5, ha="center", va="center")
    ax.text((T_opt + T[-1]) / 2, P_max_norm * 0.45,
            "Over-coupling\n(threshold too high)",
            color=CORAL, fontsize=9.5, ha="center", va="center")

    # T_opt annotation
    ax.annotate(rf"$T_\mathrm{{opt}} = {T_opt:.3f}$",
                xy=(T_opt, 0.005), xytext=(T_opt + 0.015, P_max_norm * 0.18),
                fontsize=10, color=CORAL,
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))

    # P_max annotation
    ax.text(0.002, P_max_norm + 0.003,
            r"$P_\mathrm{max} \propto (\sqrt{g}-\sqrt{\ell})^2$",
            color=CORAL, fontsize=10, va="bottom")

    ax.set_xlim(0, T[-1] + 0.005)
    ax.set_ylim(0, P_max_norm * 1.30)
    ax.set_xlabel(r"Mirror Transmissivity $T = 1 - R$", fontsize=13)
    ax.set_ylabel(r"Output Power $P_\mathrm{out}\;/\;(P_s/2)$", fontsize=13)
    ax.set_title("Optimal Mirror Transmissivity for Maximum Output Power", fontsize=15, pad=12)
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=1,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "output_power_vs_transmission.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_mode_selection_at_turnon()
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
    gm_x0, gm_x1 = 3.0, 7.0
    gm_y0, gm_y1 = 1.2, 3.6
    rect = mpatches.FancyBboxPatch((gm_x0, gm_y0), gm_x1 - gm_x0, gm_y1 - gm_y0,
                                   boxstyle="round,pad=0.05",
                                   facecolor=TEAL, alpha=0.12,
                                   edgecolor=TEAL, linewidth=1.8, zorder=2)
    ax.add_patch(rect)
    ax.text((gm_x0 + gm_x1) / 2, 2.8,
            r"Gain Medium $\quad (\gamma,\;\alpha_s)$",
            ha="center", va="center", fontsize=12, color=TEAL)

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
    draw_path(m1_x, m2_x - R, y1, TEAL,
              r"$\times\;e^{-j\beta L}\,e^{(\gamma-\alpha_s)L/2}$", label_above=True)
    
    ax.plot(m1_x, y1, 'o', color=TEAL, markersize=6, zorder=5)
    ax.text(m1_x + 0.1, y1 + 0.12, r"$E_1$", ha="left", va="bottom", fontsize=12, color=TEAL)

    # ── Stage 2: reflection off M2 → E3 ─────────────────────────────────────
    arc_right_top = mpatches.Arc((m2_x - R, (y1 + y2) / 2), 2*R, y1 - y2,
                                 angle=0, theta1=0, theta2=90,
                                 color=TEAL, lw=2.2, zorder=3)
    arc_right_bot = mpatches.Arc((m2_x - R, (y1 + y2) / 2), 2*R, y1 - y2,
                                 angle=0, theta1=-90, theta2=0,
                                 color=CORAL, lw=2.2, zorder=3)
    ax.add_patch(arc_right_top)
    ax.add_patch(arc_right_bot)
    
    # Dot and labels at the apex touching M2
    ax.plot(m2_x, (y1 + y2) / 2, 'o', color=TEAL, markersize=6, zorder=5)
    ax.text(m2_x - 0.1, (y1 + y2) / 2 + 0.2, r"$E_2$", ha="right", va="center", fontsize=12, color=TEAL)
    ax.text(m2_x - 0.1, (y1 + y2) / 2 - 0.2, r"$E_3$", ha="right", va="center", fontsize=12, color=CORAL)
    ax.text(m2_x - R / 2, (y1 + y2) / 2, r"$\times\,r_2$", ha="center", va="center", fontsize=11, color=CORAL)

    # ── Stage 3: E3 → leftward → E4 (at M1) ────────────────────────────────
    draw_path(m2_x - R, m1_x + R, y2, CORAL,
              r"$\times\;e^{-j\beta L}\,e^{(\gamma-\alpha_s)L/2}$", label_above=False)

    # ── Stage 4: reflection off M1 → E5 ─────────────────────────────────────
    arc_left_top = mpatches.Arc((m1_x + R, (y2 + y3) / 2), 2*R, y2 - y3,
                                angle=0, theta1=90, theta2=180,
                                color=CORAL, lw=2.2, zorder=3)
    arc_left_bot = mpatches.Arc((m1_x + R, (y2 + y3) / 2), 2*R, y2 - y3,
                                angle=0, theta1=180, theta2=270,
                                color=LAVENDER, lw=2.2, zorder=3)
    ax.add_patch(arc_left_top)
    ax.add_patch(arc_left_bot)
    
    # Dot and labels at the apex touching M1
    ax.plot(m1_x, (y2 + y3) / 2, 'o', color=CORAL, markersize=6, zorder=5)
    ax.text(m1_x + 0.1, (y2 + y3) / 2 + 0.2, r"$E_4$", ha="left", va="center", fontsize=12, color=CORAL)
    ax.text(m1_x + 0.1, (y2 + y3) / 2 - 0.2, r"$E_5$", ha="left", va="center", fontsize=12, color=LAVENDER)
    ax.text(m1_x + R / 2, (y2 + y3) / 2, r"$\times\,r_1$", ha="center", va="center", fontsize=11, color=LAVENDER)

    # ── Stage 5: E5 → rightward (stub) ──────────────────────────────────────
    draw_path(m1_x + R, m1_x + 3.0, y3, LAVENDER, label="", label_above=True)

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
# Plot 8: Gain Saturation Curve (Steady-State)
# Concept: As the pump rate R (or equivalently gamma_0) increases above the
# threshold alpha_r, the small-signal gain exceeds alpha_r. But once oscillation
# starts, the growing intracavity flux saturates the gain back down to alpha_r
# exactly. Below threshold: gamma = gamma_0. Above threshold: gamma locked to alpha_r.
# ─────────────────────────────────────────────────────────────────────────────
def plot_gain_saturation_curve():
    alpha_r = 0.4
    # x-axis: normalised pump parameter r = R / R_th (or equivalently gamma_0 / alpha_r)
    r = np.linspace(0, 3.0, 800)

    # Small-signal gain grows linearly with pump (3-level result)
    gamma0 = alpha_r * r   # = alpha_r when r=1

    # Clamped (saturated) gain: below threshold it equals gamma0; above, locks to alpha_r
    gamma_sat = np.where(r <= 1.0, gamma0, np.full_like(r, alpha_r))

    # Intracavity photon flux = Phi_s * (gamma_0/alpha_r - 1) for r>1, else 0
    Phi_norm = np.where(r > 1.0, (r - 1.0), 0.0)   # Phi / Phi_s

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Gain curves
    ax1.plot(r, gamma0,   color=SKYBLUE, linewidth=2.5, linestyle="--",
             label=r"Small-signal gain $\gamma_0$")
    ax1.plot(r, gamma_sat, color=TEAL, linewidth=2.5,
             label=r"Saturated gain $\gamma$ (steady state)")
    ax1.axhline(alpha_r, color=CORAL, linewidth=1.8, linestyle=":",
                label=r"Cavity loss $\alpha_r$")

    # Threshold marker
    ax1.axvline(1.0, color=GOLD, linewidth=1.4, linestyle=":")
    ax1.text(1.03, 0.05, r"Threshold  $R = R_\mathrm{th}$",
             color=GOLD, fontsize=10, va="bottom")

    # Shade above-threshold region
    ax1.axvspan(1.0, r[-1], color=TEAL, alpha=0.06)
    ax1.text(2.0, 0.09, "Oscillation\nregime", color=TEAL,
             fontsize=9.5, ha="center")

    # Second y-axis: intracavity flux
    ax2 = ax1.twinx()
    ax2.plot(r, Phi_norm, color=LAVENDER, linewidth=2.2, linestyle="-.",
             label=r"Intracavity flux $\Phi/\Phi_s$")
    ax2.set_ylabel(r"Normalised Intracavity Flux $\Phi\,/\,\Phi_s$",
                   fontsize=12, color=LAVENDER)
    ax2.tick_params(axis="y", colors=LAVENDER)
    ax2.set_ylim(-0.1, 2.2)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left",
               framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc")

    ax1.set_xlim(0, 3.0)
    ax1.set_ylim(-0.02, alpha_r * 1.6)
    ax1.set_xlabel(r"Normalised Pump Rate $R\,/\,R_\mathrm{th}$", fontsize=13)
    ax1.set_ylabel(r"Gain / Loss Coefficient", fontsize=13)
    ax1.set_yticks([0, alpha_r])
    ax1.set_yticklabels(["0", r"$\alpha_r$"], fontsize=11)
    ax1.set_title("Gain Clamping at Threshold: Steady-State Saturation", fontsize=15, pad=12)
    ax1.grid(True)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "gain_saturation_curve.jpg")
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
    nu_HB    = np.linspace(-35, 35, 2000)
    g_HB_unsat = gamma0_H / (1 + (2 * (nu_HB - nu0) / delta_H) ** 2)

    # Saturation factor: whole curve sinks until peak touches alpha_r
    sat_H = alpha_r / gamma0_H
    g_HB_sat = g_HB_unsat * sat_H

    # IHB: Gaussian gain (unsaturated envelope remains largely intact)
    sigma_I  = 9.0
    gamma0_I = 1.0
    nu_IHB   = np.linspace(-35, 35, 2000)
    g_IHB    = gamma0_I * np.exp(-0.5 * (nu_IHB / sigma_I) ** 2)

    # Discrete modes
    modes = np.arange(-32, 33, nu_F)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

    # ── Left panel: Homogeneous Broadening ───────────────────────────────────
    ax1.plot(nu_HB, g_HB_unsat, color=TEAL, linewidth=2.2, linestyle="--",
             alpha=0.6, label=r"$\gamma_0(\nu)$ — unsaturated")
    ax1.plot(nu_HB, g_HB_sat,   color=TEAL, linewidth=2.5,
             label=r"$\gamma(\nu)$ — globally saturated")
    ax1.axhline(alpha_r, color=CORAL, linewidth=1.8, linestyle="--",
                label=r"$\alpha_r$")

    # Only the mode at nu0 survives
    for m in modes:
        g_local = gamma0_H * sat_H / (1 + (2 * (m - nu0) / delta_H) ** 2)
        col  = TEAL if abs(m - nu0) < nu_F / 2 else AXES_CLR
        alph = 0.9 if abs(m - nu0) < nu_F / 2 else 0.2
        ax1.plot([m, m], [0, g_local], color=col, linewidth=2.0, alpha=alph)
        ax1.plot(m, g_local, 'o', color=col, markersize=5, alpha=alph)

    ax1.text(0, -0.06, r"Single lasing mode at $\nu_0$",
             ha="center", va="top", fontsize=10, color=TEAL)
    ax1.set_title("Homogeneous Broadening\n(Global Saturation → Single Mode)",
                  fontsize=13, pad=8)
    ax1.set_xlim(-32, 32)
    ax1.set_ylim(-0.08, 1.20)
    ax1.set_xticks([nu0])
    ax1.set_xticklabels([r"$\nu_0$"], fontsize=12)
    ax1.set_yticks([0, alpha_r, gamma0_H])
    ax1.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0^{\max}$"], fontsize=10)
    ax1.set_xlabel(r"Optical Frequency $\nu$", fontsize=12)
    ax1.set_ylabel(r"Gain / Mode Power", fontsize=12)
    ax1.grid(True)
    ax1.legend(fontsize=9.5, loc="upper right",
               framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc")

    # ── Right panel: Inhomogeneous Broadening ─────────────────────────────────
    ax2.plot(nu_IHB, g_IHB, color=SKYBLUE, linewidth=2.5, linestyle="--",
             label=r"$\gamma_0(\nu)$ — Gaussian envelope")
    ax2.axhline(alpha_r, color=CORAL, linewidth=1.8, linestyle="--",
                label=r"$\alpha_r$")

    # All modes within bandwidth B survive independently
    surviving_modes = [m for m in modes
                       if gamma0_I * np.exp(-0.5 * (m / sigma_I) ** 2) > alpha_r]
    for m in modes:
        g_local = gamma0_I * np.exp(-0.5 * (m / sigma_I) ** 2)
        alive = g_local > alpha_r
        col  = SKYBLUE if alive else AXES_CLR
        alph = 0.9 if alive else 0.2
        ax2.plot([m, m], [0, g_local], color=col, linewidth=2.0, alpha=alph)
        ax2.plot(m, g_local, 'o', color=col, markersize=5, alpha=alph)

    # B annotation
    nu_lo_I = -sigma_I * np.sqrt(-2 * np.log(alpha_r / gamma0_I))
    nu_hi_I =  sigma_I * np.sqrt(-2 * np.log(alpha_r / gamma0_I))
    ax2.annotate("", xy=(nu_lo_I, alpha_r + 0.10),
                 xytext=(nu_hi_I, alpha_r + 0.10),
                 arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.0))
    ax2.text(0, alpha_r + 0.14, r"Gain Bandwidth $B$  ($\sim N = B/\nu_F$ modes)",
             ha="center", va="bottom", fontsize=9.5, color=GOLD)

    ax2.set_title("Inhomogeneous Broadening\n(Spectral Hole Burning → Multi-Mode)",
                  fontsize=13, pad=8)
    ax2.set_xlim(-32, 32)
    ax2.set_ylim(-0.08, 1.20)
    ax2.set_xticks([nu0])
    ax2.set_xticklabels([r"$\nu_0$"], fontsize=12)
    ax2.set_yticks([0, alpha_r, gamma0_I])
    ax2.set_yticklabels(["0", r"$\alpha_r$", r"$\gamma_0^{\max}$"], fontsize=10)
    ax2.set_xlabel(r"Optical Frequency $\nu$", fontsize=12)
    ax2.grid(True)
    ax2.legend(fontsize=9.5, loc="upper right",
               framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.suptitle("Steady-State Mode Spectrum: HB vs. IHB Laser",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "hb_vs_ihb_mode_spectrum.jpg")
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
    # New 6
    plot_fp_cavity_round_trip()
    plot_airy_function_spectrum()
    plot_gain_saturation_curve()
    plot_lamb_dip()
    plot_fp_gain_ripple_spectrum()
    plot_hb_vs_ihb_mode_spectrum()
