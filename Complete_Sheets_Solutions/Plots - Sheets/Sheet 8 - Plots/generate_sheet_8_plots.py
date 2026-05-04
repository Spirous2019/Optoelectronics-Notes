"""
generate_sheet_8_plots.py
=========================
Generates all matplotlib figures for Sheet 8 of the Optoelectronics course.

Produces:
  - q6_power_distribution.jpg : optical power P(x) vs depth inside the
                                  Si photodetector (Q6a sketch).

Run from any working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

# ── Paths ────────────────────────────────────────────────────────────────────
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
#  FIGURE 1 — Power Distribution P(x) Inside Si Photodetector   [Q6(a)]
# ═══════════════════════════════════════════════════════════════════════════

def plot_power_distribution():
    """
    Parameters (from Q6):
      n_si  = 3.5,  n_air = 1.0
      alpha = 800 cm^-1 = 8e4 m^-1
      x0    = 5  µm  (P+ contact layer)
      W     = 10 µm  (depletion / intrinsic region)
      P_inc = 1.5 µW

    Layout (x-axis in µm, origin at air–Si surface):
      x < 0      : air
      0 ≤ x < x0 : P+ contact layer   (absorbing, no collection)
      x0 ≤ x < x0+W : depletion region  (absorbing, carriers collected)
      x > x0+W   : substrate (ignored)
    """
    # ── Device parameters ────────────────────────────────────────────────
    n_si   = 3.5
    n_air  = 1.0
    alpha  = 8e4        # m^-1
    x0     = 5e-6       # m
    W      = 10e-6      # m
    P_inc  = 1.5e-6     # W

    R = ((n_si - n_air) / (n_si + n_air))**2      # ≈ 0.3086
    P_trans = (1 - R) * P_inc                     # power entering Si at x=0+

    # key depths in µm for plotting
    x_um = np.linspace(-3, 20, 3000)              # µm
    x_m  = x_um * 1e-6

    # power profile:  step down at x=0 (reflection), then exp decay for x>0
    P = np.where(
        x_m < 0,
        P_inc,
        P_trans * np.exp(-alpha * np.where(x_m < 0, 0, x_m))
    ) * 1e6   # convert to µW for axis

    # ── Key power levels ─────────────────────────────────────────────────
    P_enter  = P_trans * 1e6                              # at x=0+  [µW]
    P_at_x0  = P_trans * np.exp(-alpha * x0) * 1e6       # at x=x0
    P_at_end = P_trans * np.exp(-alpha * (x0 + W)) * 1e6 # at x=x0+W
    P_inc_uW = P_inc * 1e6

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    # Main power curve
    ax.plot(x_um, P, color=TEAL, linewidth=2.5, label=r"$P(x)$  (optical power)")

    # Discontinuous step at x=0 (surface reflection)
    ax.plot([0, 0], [P_inc_uW, P_enter],
            color=CORAL, linewidth=2.0, linestyle="--", zorder=5)
    ax.annotate(r"$\Delta P = R\,P_{\rm inc}$  (reflected)",
                xy=(0, (P_inc_uW + P_enter) / 2),
                xytext=(3.5, (P_inc_uW + P_enter) / 2 + 0.05),
                fontsize=10, color=CORAL,
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0))

    # Vertical boundary lines
    ax.axvline(0,          color=AXES_CLR, linewidth=1.2, linestyle=":")
    ax.axvline(x0*1e6,     color=GOLD,     linewidth=1.5, linestyle="--",
               label=r"Start of depletion region  ($x = x_0 = 5\,\mu$m)")
    ax.axvline((x0+W)*1e6, color=LAVENDER, linewidth=1.5, linestyle="--",
               label=r"End of depletion region  ($x = x_0 + W = 15\,\mu$m)")

    # Horizontal annotation of key power levels
    for yval, label, clr in [
        (P_inc_uW,  r"$P_{\rm inc} = 1.5\,\mu$W",                          SKYBLUE),
        (P_enter,   r"$(1-R)P_{\rm inc} = 1.037\,\mu$W",                   TEAL),
        (P_at_x0,   r"$P(x_0) = 0.695\,\mu$W",                             GOLD),
        (P_at_end,  r"$P(x_0+W) = 0.312\,\mu$W",                           LAVENDER),
    ]:
        ax.axhline(yval, color=clr, linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(19.5, yval + 0.01, label,
                fontsize=9, color=clr, ha="right", va="bottom")

    # Shaded regions
    ax.axvspan(-3, 0,            alpha=0.06, color=SKYBLUE,  label="Air")
    ax.axvspan(0, x0*1e6,        alpha=0.12, color=CORAL,    label=r"P$^+$ contact layer (inactive)")
    ax.axvspan(x0*1e6, (x0+W)*1e6, alpha=0.12, color=TEAL,  label="Depletion region (active)")

    # Dot markers at boundaries
    for xv, yv, clr in [
        (0,              P_enter,   TEAL),
        (x0*1e6,         P_at_x0,   GOLD),
        ((x0+W)*1e6,     P_at_end,  LAVENDER),
    ]:
        ax.plot(xv, yv, 'o', color=clr, markersize=7, zorder=6)

    ax.set_xlim(-3, 20)
    ax.set_ylim(0, P_inc_uW * 1.18)
    ax.set_xlabel(r"Depth $x$  ($\mu$m)", fontsize=13)
    ax.set_ylabel(r"Optical Power $P(x)$  ($\mu$W)", fontsize=13)
    ax.set_title(
        r"Power Distribution Inside a Silicon p-i-n Photodetector  ($\lambda = 0.8\,\mu$m, "
        r"$\alpha = 800\,\text{cm}^{-1}$)",
        fontsize=13, pad=10
    )
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=1, facecolor="#f5f5f5",
              edgecolor="#cccccc", fontsize=9, ncol=1)

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "q6_power_distribution.jpg")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_power_distribution()
    print("All Sheet 8 plots generated.")
