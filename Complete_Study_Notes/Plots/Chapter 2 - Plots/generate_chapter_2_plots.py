"""
generate_chapter_2_plots.py
===========================
Generates matplotlib figures for Chapter 2 of the Optoelectronics notes.

Currently produces:
  - boltzmann_distribution.jpg : Shows the exponential decay curve of thermal population
                                 vs energy, highlighting that N1 >> N2.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR  # Image will be saved next to script as requested

# ── Colour palette ───────────────────────────────
WHITE    = "#ffffff"
AXES_CLR = "#2b2b2b"
GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"
CORAL    = "#d94452"
LAVENDER = "#7e57c2"

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

def plot_boltzmann_distribution():
    # E_T = k_B T (Normalised to 1 for generic plot)
    E = np.linspace(0.0, 5.0, 2000)
    
    def population(e):
        return np.exp(-e)
        
    N = population(E)
    
    # ── Pick states ──
    E1 = 0.8
    E2 = 3.2
    N1 = population(E1)
    N2 = population(E2)
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    

    # Main exponential curve
    ax.plot(E, N, color=TEAL, linewidth=2.5, label=r"Boltzmann Factor $e^{-E / k_B T}$")
    
    # Dots for states E1 and E2
    ax.plot(E1, N1, 'o', color=CORAL, markersize=8, zorder=5)
    ax.plot(E2, N2, 'o', color=CORAL, markersize=8, zorder=5)
    
    # Reference dotted lines to axes
    ax.hlines(y=[N1, N2], xmin=0, xmax=[E1, E2], color=CORAL, linestyles=":", linewidth=1.5, zorder=3)
    ax.vlines(x=[E1, E2], ymin=0, ymax=[N1, N2], color=CORAL, linestyles=":", linewidth=1.5, zorder=3)
    
    # Annotations indicating N1 >> N2
    ax.text(E1 + 0.15, N1 + 0.04, r"Lower State ($E_1$): Populated",
            fontsize=12, color=CORAL, ha="left", va="bottom")
                
    ax.text(E2, N2 + 0.12, r"Upper State ($E_2$): Barely Populated",
            fontsize=12, color=CORAL, ha="center", va="bottom")
    
    # ── Cosmetic axes formatting ──
    ax.set_xlim(0, 5.0)
    ax.set_ylim(0, 1.05)
    
    ax.set_xticks([0, E1, E2])
    ax.set_xticklabels([r"$0$", r"$E_1$", r"$E_2$"], fontsize=12)
    
    y_ticks = [0, N2, N1]
    y_labels = [r"$0$", r"$N_2$", r"$N_1$"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)
    
    ax.set_xlabel(r"Energy Level $E$", fontsize=13)
    ax.set_ylabel(r"Relative Population Density $N(E)$", fontsize=13)
    ax.set_title("Thermal Equilibrium: The Boltzmann Population Distribution", fontsize=15, pad=12)
    ax.grid(True)
    
    # ── Legend ──
    handles = [
        plt.Line2D([0], [0], color=TEAL, linewidth=2.5, label=r"$N(E) \propto e^{-E/k_B T}$ (Boltzmann distribution)")
    ]
    # Place legend in upper right (empty space for an decaying curve)
    ax.legend(handles=handles, loc="upper right", framealpha=1, facecolor="#f5f5f5", edgecolor="#cccccc", labelcolor=AXES_CLR)
    
    fig.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "boltzmann_distribution.jpg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    plot_boltzmann_distribution()
