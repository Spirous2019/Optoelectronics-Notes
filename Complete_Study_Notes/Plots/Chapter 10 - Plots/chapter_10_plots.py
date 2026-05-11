"""
Chapter 10 Plots — Photodetectors and Noise
Python figures: 2, 3, 7, 9, 11, 12
AI-generated figures: 1 (absorber slab), 4 (PN collection zones), 5 (drift schematic),
                      6 (diffusion schematic), 8 (hetero-PIN), 10 (APD composite)
— see Chapter 10 Plots.md for AI prompts.
Output: Chapter 10 - Plots/ (same directory as this script)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

WHITE    = "#ffffff"; AXES_CLR = "#2b2b2b"; GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"; CORAL    = "#d94452"; GOLD     = "#c28800"
LAVENDER = "#7e57c2"; SKYBLUE  = "#1976d2"; ORANGE   = "#e65100"
MINT     = "#00897b"; PINK     = "#ad1457"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.facecolor": WHITE, "figure.facecolor": WHITE,
    "axes.edgecolor": AXES_CLR, "axes.labelcolor": AXES_CLR,
    "xtick.color": AXES_CLR, "ytick.color": AXES_CLR, "text.color": AXES_CLR,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "grid.color": GRID_CLR, "grid.linestyle": ":", "grid.linewidth": 0.6,
    "grid.alpha": 0.7, "lines.linewidth": 2.5,
})

HERE = os.path.dirname(os.path.abspath(__file__))

def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ==============================================================================
#  FIG-2  Absorption Coefficient vs. Wavelength for Detector Materials
# ==============================================================================
def fig_02():
    lam = np.linspace(0.25, 1.80, 800)  # um

    def alpha_material(lam, Eg_eV, A=5e4, direct=True):
        """Simple model: alpha ~ A*sqrt(hnu - Eg) for direct, power law for indirect."""
        hnu = 1.24 / lam
        if direct:
            alpha = A * np.sqrt(np.maximum(hnu - Eg_eV, 0))
        else:
            alpha = A * np.maximum(hnu - Eg_eV, 0)**2
        return np.clip(alpha, 1, 1e6)

    # Material parameters: (Eg eV, direct?, scale, label, color, lw)
    materials = [
        (1.124, False, 2e4,  "Si",       AXES_CLR, 2.2),
        (0.661, False, 5e4,  "Ge",       SKYBLUE,  2.2),
        (1.424, True,  8e4,  "GaAs",     TEAL,     2.2),
        (1.350, True,  6e4,  "InP",      CORAL,    2.2),
        (0.750, True,  1e5,  "InGaAs\n(1.3–1.55 µm)", GOLD, 2.5),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for (Eg, direct, A, lbl, clr, lw) in materials:
        alpha = alpha_material(lam, Eg, A=A, direct=direct)
        ax.semilogy(lam, alpha, color=clr, lw=lw, label=lbl)

        # Direct label near cutoff
        lam_cutoff = 1.24 / Eg
        if lam_cutoff < 1.75:
            ax.text(lam_cutoff + 0.03, max(alpha)*0.25, lbl,
                    color=clr, fontsize=9.5, va="center")

    # Telecom window highlight
    ax.axvspan(1.30, 1.55, color=GOLD, alpha=0.10)
    ax.text(1.425, 3e5, "Telecom\nwindow", ha="center", color=GOLD,
            fontsize=9.5, fontweight="bold")

    # Secondary x-axis: photon energy
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    lam_ticks = [0.4, 0.5, 0.7, 1.0, 1.3, 1.6]
    ax2.set_xticks(lam_ticks)
    ax2.set_xticklabels([f"{1.24/l:.2f}" for l in lam_ticks], fontsize=9)
    ax2.set_xlabel(r"Photon energy  (eV)", fontsize=11, color=AXES_CLR)
    ax2.tick_params(colors=AXES_CLR)

    ax.set_xlabel(r"Wavelength  ($\mu$m)", fontsize=13)
    ax.set_ylabel(r"Absorption coefficient $\alpha$  (cm$^{-1}$)", fontsize=13)
    ax.set_title("Absorption Coefficient vs. Wavelength — Detector Materials", pad=12)
    ax.set_xlim(0.25, 1.80); ax.set_ylim(10, 2e6)
    ax.legend(loc="upper right", framealpha=0.9, facecolor="#f5f5f5",
              edgecolor="#cccccc", fontsize=10)
    ax.grid(True)

    fig.tight_layout()
    save(fig, "absorption_coefficient.jpg")


fig_02()


# ==============================================================================
#  FIG-3  Quantum Efficiency and Responsivity vs. Wavelength (Two-Panel)
# ==============================================================================
def fig_03():
    lam = np.linspace(0.4, 1.8, 600)   # um; InGaAs-based detector

    # QE: trapezoid — low below ~0.7 (surface losses), plateau 0.7–1.6, drop at 1.65
    def eta(lam):
        lo_edge = 0.7; hi_edge = 1.65
        peak_eta = 0.85
        rise  = np.clip((lam - 0.4) / (lo_edge - 0.4), 0, 1)
        fall  = np.clip((hi_edge - lam) / 0.10, 0, 1)
        return peak_eta * rise * fall

    eta_arr = eta(lam)

    # Responsivity R = eta * e * lambda / (hc) = eta * lambda / 1.24  (A/W, lambda in um)
    R_arr = eta_arr * lam / 1.24

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ── Top: QE ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(lam, 0, eta_arr, color=SKYBLUE, alpha=0.18)
    ax.plot(lam, eta_arr, color=SKYBLUE, lw=2.5)
    ax.axvline(1.65, color=AXES_CLR, lw=1.0, ls="--")
    ax.text(1.67, 0.70, r"$\lambda_{cut}$", color=AXES_CLR, fontsize=10)
    ax.set_ylabel(r"Quantum efficiency $\eta(\lambda)$", fontsize=12)
    ax.set_title(r"(a) Quantum Efficiency", pad=8)
    ax.set_xlim(0.4, 1.8); ax.set_ylim(0, 1.10)
    ax.grid(True)

    # ── Bottom: Responsivity ──────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(lam, 0, R_arr, color=CORAL, alpha=0.18)
    ax.plot(lam, R_arr, color=CORAL, lw=2.5)
    ax.axvline(1.65, color=AXES_CLR, lw=1.0, ls="--")

    # Ideal responsivity line (100% QE)
    R_ideal = lam / 1.24
    ax.plot(lam, R_ideal, color=AXES_CLR, lw=1.3, ls=":", alpha=0.6,
            label=r"Ideal $\eta = 1$: $\mathcal{R} = \lambda/1.24$")
    ax.text(1.5, R_arr.max()*0.60,
            r"$\mathcal{R} = \dfrac{\eta e\lambda}{hc}$",
            color=CORAL, fontsize=11)

    ax.set_xlabel(r"Wavelength $\lambda$  ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Responsivity $\mathcal{R}$  (A/W)", fontsize=12)
    ax.set_title(r"(b) Responsivity", pad=8)
    ax.set_ylim(0, R_ideal.max()*1.10)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout(h_pad=0.8)
    save(fig, "qe_and_responsivity.jpg")


fig_03()


# ==============================================================================
#  FIG-7  PIN Photodiode: Layer Structure, Charge Density, and Electric Field
# ==============================================================================
def fig_07():
    x = np.linspace(0, 5.0, 600)   # um; p|i|n from 0 to 5

    Xp  = 0.4   # p-layer thickness
    Xi1 = Xp
    Xi2 = Xp + 3.5  # intrinsic layer 3.5 um thick
    Xn2 = Xi2 + 0.4

    # Charge density rho(x)
    rho = np.where(x < Xp, -1.0,
          np.where(x < Xi1, 0.0,
          np.where(x < Xi2, 0.0,
          np.where(x < Xn2, 1.0, 0.0))))

    # Electric field E(x): integrate from rho (reverse bias)
    # Nearly flat in intrinsic region
    E = np.where(x < Xp, -(x / Xp),
        np.where(x < Xi2, -1.0,
        np.where(x < Xn2, -1.0 + (x - Xi2) / (Xn2 - Xi2), 0.0)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.5, 1.5]})

    # ── Panel A: layer structure ──────────────────────────────────────────────
    ax = axes[0]
    ax.axvspan(0,    Xp,  color=CORAL,  alpha=0.25, label="p layer")
    ax.axvspan(Xp,   Xi2, color=GOLD,   alpha=0.15, label="i layer (intrinsic)")
    ax.axvspan(Xi2,  Xn2, color=SKYBLUE,alpha=0.25, label="n layer")
    ax.axvspan(Xn2,  5.0, color=WHITE)

    ax.text((0 + Xp)/2,     0.5, "p+",  ha="center", color=CORAL,   fontsize=12, fontweight="bold")
    ax.text((Xp + Xi2)/2,   0.5, "i",   ha="center", color=AXES_CLR,fontsize=12, fontweight="bold")
    ax.text((Xi2 + Xn2)/2,  0.5, "n+",  ha="center", color=SKYBLUE, fontsize=12, fontweight="bold")

    # Reverse bias annotation
    ax.annotate("", xy=(Xn2, 0.9), xytext=(0, 0.9),
                arrowprops=dict(arrowstyle="<-", color=AXES_CLR, lw=1.4))
    ax.text(Xi2/2 + 0.2, 0.95, r"$V_{reverse}$", color=AXES_CLR, fontsize=10, ha="center")

    ax.set_yticks([]); ax.set_ylim(0, 1.1)
    ax.set_title("(a) PIN Structure", pad=6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(False)

    # ── Panel B: charge density ───────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(x, 0, rho, where=(rho > 0), color=SKYBLUE, alpha=0.30)
    ax.fill_between(x, 0, rho, where=(rho < 0), color=CORAL,   alpha=0.30)
    ax.plot(x, rho, color=AXES_CLR, lw=2.5)
    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.axvline(Xp,  color=AXES_CLR, lw=0.9, ls=":")
    ax.axvline(Xi2, color=AXES_CLR, lw=0.9, ls=":")
    ax.set_ylabel(r"Charge density $\rho(x)$", fontsize=12)
    ax.set_title("(b) Net Charge Density", pad=6)
    ax.set_ylim(-1.4, 1.4); ax.set_yticks([])
    ax.text(Xp/2, 1.1, r"$-$", ha="center", color=CORAL,  fontsize=14)
    ax.text((Xi2+Xn2)/2, 1.1, r"$+$", ha="center", color=SKYBLUE, fontsize=14)
    ax.grid(False)

    # ── Panel C: electric field ───────────────────────────────────────────────
    ax = axes[2]
    ax.fill_between(x, 0, E, color=TEAL, alpha=0.20)
    ax.plot(x, E, color=TEAL, lw=2.5)
    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.axvline(Xp,  color=AXES_CLR, lw=0.9, ls=":")
    ax.axvline(Xi2, color=AXES_CLR, lw=0.9, ls=":")
    ax.set_ylabel(r"Electric field $E(x)$", fontsize=12)
    ax.set_xlabel(r"Position  ($\mu$m)", fontsize=12)
    ax.set_title("(c) Electric Field — Nearly Uniform Across Intrinsic Region", pad=6)
    ax.set_xlim(0, 5.0); ax.set_ylim(-1.40, 0.30); ax.set_yticks([])
    ax.annotate("Nearly flat", xy=(Xi2/2 + 0.5, -1.0), xytext=(Xi2/2 + 0.5, -0.60),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2),
                color=TEAL, fontsize=10, ha="center")
    ax.grid(False)

    fig.tight_layout(h_pad=0.5)
    save(fig, "pin_structure_and_field.jpg")


fig_07()


# ==============================================================================
#  FIG-9  Load-Line I-V Construction for a Photodiode
# ==============================================================================
def fig_09():
    V = np.linspace(-5.5, 0.8, 600)
    VB = 5.0; RL = 3.0   # V and MOhm (normalised)
    I_sat = 0.1           # dark reverse saturation

    def diode_IV(V, Ip=0):
        """Ideal diode + photocurrent: I = I_sat*(exp(V/0.026)-1) - Ip"""
        return I_sat * (np.exp(np.clip(V/0.026, -50, 15)) - 1) - Ip

    # Load line: V = VB - I*RL  =>  I = (VB - V)/RL — but we plot I vs V
    V_load = np.linspace(-VB, 0.8, 200)
    I_load = (VB + V_load) / RL   # positive current in load direction

    # Three illumination levels
    Ip_vals  = [0.0, 0.8, 1.6]
    colors9  = [AXES_CLR, TEAL, CORAL]
    labels9  = ["Dark", r"$I_p$", r"$2I_p$"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Left: circuit sketch (Matplotlib axes-art) ────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("(a) Circuit", pad=8)

    # Wire rectangle
    lw_c = 2.0
    ax.plot([1,5,5,1,1], [5,5,1,1,5], color=AXES_CLR, lw=lw_c)

    # Photodiode symbol (triangle + line)
    ax.fill([2.8,3.2,3.0], [3.8,3.8,3.2], color=AXES_CLR)
    ax.plot([3.0,3.0], [3.2,3.8], color=AXES_CLR, lw=lw_c)
    ax.plot([2.8,3.2], [3.2,3.2], color=AXES_CLR, lw=lw_c)
    ax.text(3.3, 3.5, "Photodiode", color=AXES_CLR, fontsize=8.5, va="center")

    # Light arrow into diode
    ax.annotate("", xy=(3.0, 3.62), xytext=(3.7, 4.30),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
    ax.text(3.90, 4.45, r"$h\nu$", color=GOLD, fontsize=10)

    # Resistor symbol
    ax.add_patch(mpatches.FancyBboxPatch((1.5, 4.7), 1.0, 0.45,
                 boxstyle="round,pad=0.05",
                 facecolor="#f0f0f0", edgecolor=AXES_CLR, lw=lw_c))
    ax.text(2.0, 4.93, r"$R_L$", ha="center", va="center", color=AXES_CLR, fontsize=11)

    # Bias source symbol
    ax.plot([4.7,5.3], [2.6,2.6], color=AXES_CLR, lw=lw_c)
    ax.plot([4.8,5.2], [2.2,2.2], color=AXES_CLR, lw=lw_c)
    ax.text(5.3, 2.45, r"$V_B$", color=AXES_CLR, fontsize=11)

    # Current label
    ax.annotate("", xy=(3.0, 5.0), xytext=(2.5, 5.0),
                arrowprops=dict(arrowstyle="->", color=SKYBLUE, lw=1.4))
    ax.text(2.75, 5.22, r"$I$", color=SKYBLUE, fontsize=11, ha="center")

    # ── Right: load-line I-V graph ────────────────────────────────────────────
    ax = axes[1]

    for Ip, clr, lbl in zip(Ip_vals, colors9, labels9):
        I_diode = diode_IV(V, Ip=Ip)
        ax.plot(V, I_diode, color=clr, lw=2.2, label=lbl)

    # Load line
    ax.plot(V_load, I_load, color=SKYBLUE, lw=2.2, ls="--",
            label=r"Load line  $(-1/R_L)$")

    # Operating points (intersections)
    for Ip, clr in zip(Ip_vals, colors9):
        I_d = diode_IV(V, Ip=Ip)
        I_l = (VB + V) / RL
        diff = np.abs(I_d - I_l)
        idx  = np.argmin(diff)
        ax.plot(V[idx], I_d[idx], "o", ms=8, color=clr, zorder=6)

    # Axis intercepts
    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.axvline(0, color=AXES_CLR, lw=0.8)
    ax.text(-VB + 0.1, 0.04, r"$-V_B$", color=AXES_CLR, fontsize=9.5)

    ax.set_xlabel(r"Diode voltage $V$  (V, norm.)", fontsize=12)
    ax.set_ylabel(r"Current $I$  (norm.)", fontsize=12)
    ax.set_title("(b) Load-Line I–V Construction", pad=8)
    ax.set_xlim(-5.5, 0.8); ax.set_ylim(-0.3, I_load.max()*1.1)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout(pad=2.0)
    save(fig, "load_line_construction.jpg")


fig_09()


# ==============================================================================
#  FIG-11  Time-Constant Contributions to Detector Bandwidth
# ==============================================================================
def fig_11():
    contributions = [
        (r"$\tau_{drift}$",  "Drift transit\n(depletion region)", TEAL),
        (r"$\tau_{diff}$",   "Diffusion transit\n(neutral region)",  SKYBLUE),
        (r"$\tau_{RC}$",     "RC charging\n(load + parasitic)",      GOLD),
        (r"$\tau_A$",        "Avalanche\nbuild-up (APD only)",        CORAL),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(14, 4.5),
                              gridspec_kw={"width_ratios": [1,1,1,1,1.6]})
    fig.suptitle("Physical Origins of Detector Response-Time Limitation", y=1.04, fontsize=14)

    for i, (sym, desc, clr) in enumerate(contributions):
        ax = axes[i]
        ax.set_facecolor(WHITE)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        # Rounded box
        rect = mpatches.FancyBboxPatch((0.08, 0.15), 0.84, 0.70,
                    boxstyle="round,pad=0.06",
                    facecolor=clr+"22", edgecolor=clr, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(0.50, 0.68, sym,   ha="center", va="center",
                fontsize=18, color=clr, fontweight="bold")
        ax.text(0.50, 0.36, desc,  ha="center", va="center",
                fontsize=8.5, color=AXES_CLR, multialignment="center")

    # Summary box
    ax = axes[4]
    ax.set_facecolor(WHITE); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    rect2 = mpatches.FancyBboxPatch((0.05, 0.10), 0.90, 0.80,
                boxstyle="round,pad=0.06",
                facecolor="#f5f5f5", edgecolor=AXES_CLR, linewidth=2.0)
    ax.add_patch(rect2)
    ax.text(0.50, 0.80, "Effective time constant",
            ha="center", va="center", fontsize=10, color=AXES_CLR)
    ax.text(0.50, 0.55,
            r"$\tau_{eff}^2 = \tau_{drift}^2 + \tau_{diff}^2 + \tau_{RC}^2 + \tau_A^2$",
            ha="center", va="center", fontsize=9.5, color=AXES_CLR)
    ax.text(0.50, 0.30,
            r"$BW \approx \dfrac{1}{2\pi\tau_{eff}}$",
            ha="center", va="center", fontsize=12, color=SKYBLUE)

    fig.tight_layout()
    save(fig, "detector_bandwidth_contributions.jpg")


fig_11()


# ==============================================================================
#  FIG-12  APD Gain-Bandwidth Tradeoff
# ==============================================================================
def fig_12():
    M = np.linspace(1, 50, 500)

    # Gain-bandwidth product approximately constant: BW ~ C/M
    C_GBP = 30.0   # GHz (constant product)
    BW    = C_GBP / M

    # Operating region: M = 5-20 reasonable
    M_op1, M_op2 = 5, 20

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(M, BW, color=SKYBLUE, lw=2.8)

    # Constant GBP line annotation
    ax.axhline(C_GBP/M_op1,  color=AXES_CLR, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(C_GBP/M_op2, color=AXES_CLR, lw=0.8, ls="--", alpha=0.5)

    # Operating region shading
    ax.axvspan(M_op1, M_op2, color=TEAL, alpha=0.10)
    ax.text((M_op1+M_op2)/2, BW.max()*0.55,
            "Practical\noperating\nregion",
            ha="center", color=TEAL, fontsize=10, fontweight="bold")

    # GBP annotation
    ax.text(38, C_GBP/38 + 0.6,
            r"$M \cdot BW \approx \dfrac{v_d}{2\pi W_A}$",
            color=AXES_CLR, fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#cccccc", lw=0.8))

    ax.set_xlabel(r"Multiplication factor $M$", fontsize=13)
    ax.set_ylabel(r"Bandwidth $BW$  (GHz)", fontsize=13)
    ax.set_title("APD Gain–Bandwidth Tradeoff: Higher Gain Costs Bandwidth", pad=10)
    ax.set_xlim(1, 50); ax.set_ylim(0, BW.max()*1.15)
    ax.grid(True)

    fig.tight_layout()
    save(fig, "apd_gain_bandwidth.jpg")


fig_12()
