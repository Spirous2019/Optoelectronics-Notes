"""
Chapter 6 Plots — Homojunctions, Heterojunctions, and Optical Gain
All figures follow the figure-generation-style guide.
Figure 12 (optical transitions schematic) is AI-generated — see Chapter 6 Plots.md.
Output: Chapter 6 - Plots/ (same directory as this script)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

WHITE    = "#ffffff"; AXES_CLR = "#2b2b2b"; GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"; CORAL    = "#d94452"; GOLD     = "#c28800"
LAVENDER = "#7e57c2"; SKYBLUE  = "#1976d2"; MINT     = "#00897b"
ORANGE   = "#e65100"; PINK     = "#ad1457"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.facecolor": WHITE, "figure.facecolor": WHITE,
    "axes.edgecolor": AXES_CLR, "axes.labelcolor": AXES_CLR,
    "xtick.color": AXES_CLR, "ytick.color": AXES_CLR, "text.color": AXES_CLR,
    "axes.titlesize": 15, "axes.labelsize": 13,
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

def band_bend(x, hi, lo, x0=0.38, x1=0.62, s=3.5):
    """Smooth tanh S-curve band bending."""
    m = (x0 + x1) / 2; h = (x1 - x0) / 2
    return hi + (lo - hi) * 0.5 * (1 + np.tanh((x - m) / h * s))

def fermi_dirac(eps, eta, kBT):
    """Fermi-Dirac occupation for energy eps relative to a band edge."""
    return 1.0 / (1.0 + np.exp((eps - eta) / kBT))

def transition_energies(delta_e, me_eff=0.067, mh_eff=0.50):
    """Partition h*nu - Eg between electron and hole kinetic energies."""
    delta = np.maximum(delta_e, 0.0)
    frac_e = mh_eff / (me_eff + mh_eff)
    frac_h = me_eff / (me_eff + mh_eff)
    return frac_e * delta, frac_h * delta

def inversion_factor(delta_e, eta_c, eta_v, kBT, me_eff=0.067, mh_eff=0.50):
    """
    Return fc - fv for direct k-conserving transitions in bulk GaAs.

    eta_c = E_Fc - E_c
    eta_v = E_v - E_Fv
    """
    eps_e, eps_h = transition_energies(delta_e, me_eff=me_eff, mh_eff=mh_eff)
    fc = fermi_dirac(eps_e, eta_c, kBT)
    hole_occ = fermi_dirac(eps_h, eta_v, kBT)  # 1 - fv
    return fc + hole_occ - 1.0, fc, hole_occ


# ==============================================================================
#  FIG-1  p-n Homojunction Band Diagram at Thermal Equilibrium
# ==============================================================================
def fig_01():
    x = np.linspace(0, 1, 500)
    Ec_p, Ev_p = 2.2, 0.0         # p-side band edges
    Vo = 0.9                       # built-in potential
    Ec_n = Ec_p - Vo; Ev_n = Ev_p - Vo
    Ef   = (Ec_n + Ev_p) / 2      # flat Fermi level

    Ec_x = band_bend(x, Ec_p, Ec_n)
    Ev_x = band_bend(x, Ev_p, Ev_n)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(x, Ec_x, color=SKYBLUE, lw=2.5, label=r"$E_c$")
    ax.plot(x, Ev_x, color=CORAL,   lw=2.5, label=r"$E_v$")
    ax.plot([0, 1], [Ef, Ef], color=AXES_CLR, lw=1.8, ls="--", label=r"$E_F$")

    # Band shading
    ax.fill_between(x, Ec_x, Ec_x+0.55, color=SKYBLUE, alpha=0.10)
    ax.fill_between(x, Ev_x-0.55, Ev_x,  color=CORAL,   alpha=0.10)

    # Depletion boundaries
    for xd in [0.38, 0.62]:
        ax.axvline(xd, color=AXES_CLR, lw=1.0, ls=":")

    # eV0 brace
    ax.annotate("", xy=(0.85, Ec_n), xytext=(0.85, Ec_p),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.6))
    ax.text(0.87, (Ec_p+Ec_n)/2, r"$eV_o$", color=TEAL, fontsize=11, va="center")

    # Electrons on n-side, holes on p-side
    for dy in [0.08, 0.18, 0.30]:
        ax.plot(0.75+dy, Ec_n+dy+0.05, "o", color=SKYBLUE, ms=7, zorder=5)
        ax.plot(0.25-dy, Ev_p-dy-0.05, "o", mfc="white", mec=CORAL, ms=7, lw=1.5, zorder=5)

    # Region labels
    ax.text(0.17, -0.9, "p-region",        ha="center", color=CORAL,   fontsize=11)
    ax.text(0.50, -0.9, "Depletion region", ha="center", color=LAVENDER, fontsize=10)
    ax.text(0.83, -0.9, r"n$^+$-region",   ha="center", color=SKYBLUE,  fontsize=11)

    # Band edge labels
    ax.text(-0.01, Ec_p, r"$E_c$", color=SKYBLUE, fontsize=11, ha="right", va="center")
    ax.text(-0.01, Ev_p, r"$E_v$", color=CORAL,   fontsize=11, ha="right", va="center")
    ax.text(-0.01, Ef,   r"$E_F$", color=AXES_CLR, fontsize=11, ha="right", va="center")

    ax.set_xlabel(r"Distance into device $\rightarrow$", fontsize=13)
    ax.set_ylabel(r"Electron energy $E$  (a.u.)", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.05, 1.0); ax.set_ylim(-1.2, 3.0)
    ax.set_title(r"p-n Homojunction at Thermal Equilibrium", pad=10)
    ax.grid(False)
    ax.legend(loc="upper right", framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout()
    save(fig, "homojunction_equilibrium.jpg")


fig_01()


# ==============================================================================
#  FIG-2  Homojunction Under Forward Bias
# ==============================================================================
def fig_02():
    x = np.linspace(0, 1, 500)
    Ec_p, Ev_p = 2.2, 0.0
    V_applied  = 0.60              # forward bias
    Vo = 0.9
    Ec_n = Ec_p - Vo + V_applied  # reduced barrier
    Ev_n = Ev_p - Vo + V_applied

    Ec_x = band_bend(x, Ec_p, Ec_n)
    Ev_x = band_bend(x, Ev_p, Ev_n)

    # Keep the quasi-Fermi levels inside the gap on their own sides.
    Efn = Ec_n - 0.10
    Efp = Ev_p + 0.10

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(x, Ec_x, color=SKYBLUE, lw=2.5)
    ax.plot(x, Ev_x, color=CORAL,   lw=2.5)
    ax.fill_between(x, Ec_x, Ec_x+0.55, color=SKYBLUE, alpha=0.10)
    ax.fill_between(x, Ev_x-0.55, Ev_x,  color=CORAL,   alpha=0.10)

    # Quasi-Fermi levels are nearly flat in the quasi-neutral regions.
    ax.hlines(Efp, xmin=-0.05, xmax=0.38, color=CORAL,  lw=1.8, ls="--")
    ax.hlines(Efn, xmin=0.62, xmax=1.05, color=SKYBLUE, lw=1.8, ls="--")

    # Depletion boundaries
    for xd in [0.38, 0.62]:
        ax.axvline(xd, color=AXES_CLR, lw=1.0, ls=":")

    # Reduced barrier brace
    ax.annotate("", xy=(0.97, Ec_n), xytext=(0.97, Ec_p),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
    ax.text(0.99, (Ec_p+Ec_n)/2, r"$e(V_o-V)$",
            color=AXES_CLR, fontsize=9.5, va="center", ha="left")

    # Recombination wavy arrows in depletion
    for xr in [0.44, 0.50, 0.56]:
        Ec_mid = band_bend(np.array([xr]), Ec_p, Ec_n)[0]+0.10
        Ev_mid = band_bend(np.array([xr]), Ev_p, Ev_n)[0]-0.10
        ax.annotate("", xy=(xr, Ev_mid), xytext=(xr, Ec_mid),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8,
                                    connectionstyle="arc3,rad=0.3"))
    ax.text(0.50, (band_bend(np.array([0.50]),Ec_p,Ec_n)[0]
                   + band_bend(np.array([0.50]),Ev_p,Ev_n)[0])/2 - 0.05,
            r"$h\nu \approx E_g$", color=GOLD, fontsize=10, ha="center")

    # Injected carriers
    for dy in [0.06, 0.14, 0.22]:
        ax.plot(0.63+dy*0.6, Ec_n+dy, "o", color=SKYBLUE, ms=6, zorder=5)
        ax.plot(0.37-dy*0.6, Ev_p-dy, "o", mfc="white", mec=CORAL, ms=6, lw=1.4, zorder=5)

    # Labels
    ax.text(-0.01, Ec_p, r"$E_c$",    color=SKYBLUE, fontsize=11, ha="right", va="center")
    ax.text(-0.01, Ev_p, r"$E_v$",    color=CORAL,   fontsize=11, ha="right", va="center")
    ax.text(0.01, Efp, r"$E_{Fp}$", color=CORAL,   fontsize=11, ha="left", va="center")
    ax.text(0.76, Efn+0.06, r"$E_{Fn}$", color=SKYBLUE, fontsize=11, ha="left", va="bottom")

    ax.text(0.17, -0.9, "p-region",        ha="center", color=CORAL,   fontsize=11)
    ax.text(0.50, -0.9, "Recombination region", ha="center", color=GOLD, fontsize=10)
    ax.text(0.83, -0.9, r"n$^+$-region",  ha="center", color=SKYBLUE,  fontsize=11)
    ax.text(0.74, 2.28, "Split quasi-Fermi levels\ndrive carrier injection",
            ha="center", color=LAVENDER, fontsize=9.5)

    ax.set_xlabel(r"Distance into device $\rightarrow$", fontsize=13)
    ax.set_ylabel(r"Electron energy $E$  (a.u.)", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-1.2, 3.0)
    ax.set_title("Homojunction Under Forward Bias - Carrier Injection", pad=10)
    ax.grid(False)

    fig.tight_layout()
    save(fig, "homojunction_forward_bias.jpg")


fig_02()


# ==============================================================================
#  FIG-3  Depletion Region Profiles: Charge, Electric Field, Potential
# ==============================================================================
def fig_03():
    # Work entirely in normalized units so rho(x), E(x), and V(x)
    # remain self-consistent and continuous across the junction.
    Na_rel, Nd_rel = 2.0, 1.0
    ratio = Nd_rel / Na_rel
    Wno = 1.0
    Wpo = ratio * Wno

    x_p = np.linspace(-Wpo, 0, 300)
    x_n = np.linspace(0, Wno, 300)

    # rho is normalized to e*Na.
    rho_p = -np.ones_like(x_p)
    rho_n =  ratio * np.ones_like(x_n)

    # Piecewise-linear electric field from Poisson's equation.
    E_p = -(x_p + Wpo)
    E_n = -Wpo + ratio * x_n
    E_peak = abs(E_p[-1])

    # Electrostatic potential from -integral E dx, normalized to V_o.
    V_p = 0.5 * (x_p + Wpo)**2
    V_at_junction = V_p[-1]
    V_n = V_at_junction + Wpo * x_n - 0.5 * ratio * x_n**2
    V_total = V_n[-1]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.patch.set_facecolor(WHITE)

    # Shared depletion boundaries
    for ax in axes:
        for xd in [-Wpo, 0, Wno]:
            ax.axvline(xd, color=AXES_CLR, lw=1.0, ls=":")

    # Panel A: charge density
    ax = axes[0]
    ax.fill_between(x_p, 0, rho_p, color=CORAL,   alpha=0.35)
    ax.fill_between(x_n, 0, rho_n, color=SKYBLUE, alpha=0.35)
    ax.plot(x_p, rho_p, color=CORAL,   lw=2.5)
    ax.plot(x_n, rho_n, color=SKYBLUE, lw=2.5)
    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.set_ylabel(r"$\rho_\mathrm{net}/(eN_a)$", fontsize=12)
    ax.text(-Wpo*0.6, -0.55, r"$-eN_a$", color=CORAL,   fontsize=11, ha="center")
    ax.text( Wno*0.6,  0.28, r"$+eN_d$", color=SKYBLUE, fontsize=11, ha="center")
    ax.set_ylim(-1.3, 1.3); ax.set_title("(A) Net Charge Density", pad=6)
    ax.grid(True)

    # Wpo and Wno labels on panel A
    ax.annotate("", xy=(-Wpo, -1.15), xytext=(0, -1.15),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.2))
    ax.text(-Wpo/2, -1.25, r"$W_{po}$", ha="center", color=AXES_CLR, fontsize=10)
    ax.annotate("", xy=(0, -1.15), xytext=(Wno, -1.15),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.2))
    ax.text(Wno/2, -1.25, r"$W_{no}$", ha="center", color=AXES_CLR, fontsize=10)

    # Panel B: electric field
    ax = axes[1]
    ax.plot(x_p, E_p/E_peak, color=SKYBLUE, lw=2.5)
    ax.plot(x_n, E_n/E_peak, color=SKYBLUE, lw=2.5)
    ax.fill_between(np.concatenate([x_p, x_n]),
                    np.concatenate([E_p, E_n])/E_peak, 0,
                    color=SKYBLUE, alpha=0.20)
    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.set_ylabel(r"$\mathcal{E}(x)/\mathcal{E}_\mathrm{max}$", fontsize=12)
    ax.text(0, -0.55, r"$-\mathcal{E}_\mathrm{max}$", color=SKYBLUE, fontsize=11, ha="center")
    ax.set_ylim(-1.3, 0.4); ax.set_title("(B) Electric Field", pad=6)
    ax.grid(True)

    # Panel C: potential
    ax = axes[2]
    V_norm_p = V_p / V_total
    V_norm_n = V_n / V_total
    ax.plot(x_p, V_norm_p, color=CORAL, lw=2.5)
    ax.plot(x_n, V_norm_n, color=CORAL, lw=2.5)
    ax.fill_between(np.concatenate([x_p, x_n]),
                    np.concatenate([V_norm_p, V_norm_n]),
                    color=CORAL, alpha=0.15)
    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.axhline(1, color=AXES_CLR, lw=0.8, ls="--")
    ax.text(Wno*0.88, 1.06, r"$V_o$", color=CORAL, fontsize=11, va="bottom", ha="center")
    ax.set_ylabel(r"$V(x) / V_o$", fontsize=12)
    ax.set_xlabel(r"Position $x$  (norm.)", fontsize=12)
    ax.set_ylim(-0.1, 1.3); ax.set_title("(C) Electrostatic Potential", pad=6)
    ax.grid(True)

    fig.tight_layout(h_pad=0.5)
    save(fig, "depletion_profiles.jpg")


fig_03()


# ==============================================================================
#  FIG-4  Joint Density of States  rho(nu)
# ==============================================================================
def fig_04():
    Eg = 1.0   # Generic bandgap
    hnu = np.linspace(0, Eg + 1.0, 500)
    rho = np.sqrt(np.maximum(hnu - Eg, 0))   # proportional to sqrt(hnu - Eg)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(hnu, 0, rho, color=SKYBLUE, alpha=0.22)
    ax.plot(hnu, rho, color=TEAL, lw=2.5)

    # Onset at Eg
    ax.plot([Eg, Eg], [0, max(rho)*1.15], color=AXES_CLR, lw=1.2, ls="--")
    ax.text(Eg + 0.05, max(rho)*0.85,
            r"$\rho(\nu) \propto \sqrt{h\nu - E_g}$",
            color=AXES_CLR, fontsize=12)

    ax.set_xlabel(r"Photon energy $h\nu$", fontsize=13)
    ax.set_ylabel(r"Joint Density of States $\rho(\nu)$", fontsize=13)
    ax.set_title(r"Joint Density of States vs. Photon Energy", pad=10)
    
    ax.set_xlim(0, Eg + 1.0)
    ax.set_ylim(0, max(rho) * 1.15)
    
    # Generic axis values
    ax.set_xticks([0, Eg])
    ax.set_xticklabels(['$0$', '$E_g$'], fontsize=12)
    ax.set_yticks([0])
    ax.set_yticklabels(['$0$'], fontsize=12)
    
    ax.grid(True)

    fig.tight_layout()
    save(fig, "joint_dos.jpg")



fig_04()

# ==============================================================================
#  FIG-7  Minority Carrier Profile in a Forward-Biased Homojunction
# ==============================================================================
def fig_07():
    Ln, Lp = 2.0, 1.5   # diffusion lengths in um
    x_n = np.linspace(0, 6*Lp, 400)    # n-side (x>0)
    x_p = np.linspace(-6*Ln, 0, 400)   # p-side (x<0)

    delta_pn = np.exp(-x_n / Lp)   # minority holes in n
    delta_np = np.exp( x_p / Ln)   # minority electrons in p

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x_p, delta_np, color=SKYBLUE, lw=2.5, label=r"$\delta n_p$ (electrons in p)")
    ax.plot(x_n, delta_pn, color=CORAL,   lw=2.5, label=r"$\delta p_n$ (holes in n)")

    # Depletion region shading
    ax.axvspan(-0.2, 0.2, color=LAVENDER, alpha=0.18)
    ax.text(0, 0.78, "Depletion\nregion", ha="center", color=LAVENDER, fontsize=9.5)

    # Diffusion length arrows — staggered vertically to avoid collision
    ax.annotate("", xy=(-Ln, 0.50), xytext=(0, 0.50),
                arrowprops=dict(arrowstyle="<->", color=SKYBLUE, lw=1.4))
    ax.text(-Ln/2, 0.54, rf"$L_n = {Ln}\;\mu$m", ha="center", color=SKYBLUE, fontsize=10)

    ax.annotate("", xy=(Lp, 0.65), xytext=(0, 0.65),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.4))
    ax.text(Lp/2, 0.69, rf"$L_p = {Lp}\;\mu$m", ha="center", color=CORAL, fontsize=10)

    ax.set_xlabel(r"Position $x$  ($\mu$m)", fontsize=13)
    ax.set_ylabel("Normalized excess carrier density", fontsize=12)
    ax.set_title("Minority Carrier Profiles — Homojunction (Forward Bias)", pad=10)
    ax.set_xlim(-6*Ln, 6*Lp); ax.set_ylim(-0.05, 1.15)
    ax.text(-8, 0.92, "p-side", color=SKYBLUE, fontsize=11)
    ax.text(5, 0.92,  "n-side", color=CORAL,   fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "minority_carrier_profile.jpg")


fig_07()

# ==============================================================================
#  FIG-12  E-k Band Diagram (Optical Gain)
# ==============================================================================
def fig_12():
    # Parabola parameters
    Eg = 2.0
    Ec = Eg / 2
    Ev = -Eg / 2
    mc = 1.0
    mv = 1.5

    fig, ax = plt.subplots(figsize=(9, 7))
    
    k = np.linspace(-3, 3, 100)
    E_c_k = Ec + k**2 / mc
    E_v_k = Ev - k**2 / mv

    # Quasi-Fermi levels
    Efc = Ec + 3.5
    Efv = Ev - 3.5

    # Plot parabolas
    ax.plot(k, E_c_k, color=AXES_CLR, lw=2)
    ax.plot(k, E_v_k, color=AXES_CLR, lw=2)

    # Shaded regions (filled states)
    k_fill_c = np.linspace(-np.sqrt((Efc - Ec) * mc), np.sqrt((Efc - Ec) * mc), 100)
    E_c_k_fill = Ec + k_fill_c**2 / mc
    ax.fill_between(k_fill_c, E_c_k_fill, Efc, color=GRID_CLR, alpha=0.5)

    k_fill_v = np.linspace(-np.sqrt((Ev - Efv) * mv), np.sqrt((Ev - Efv) * mv), 100)
    E_v_k_fill = Ev - k_fill_v**2 / mv
    ax.fill_between(k_fill_v, Efv, E_v_k_fill, color=GRID_CLR, alpha=0.5)

    # Dots on the parabolas (electrons and holes)
    k_dots_c = np.linspace(-2.6, 2.6, 17)
    E_dots_c = Ec + k_dots_c**2 / mc
    for x, y in zip(k_dots_c, E_dots_c):
        if y <= Efc:
            ax.plot(x, y, 'o', color=AXES_CLR, ms=8)
        else:
            ax.plot(x, y, 'o', color=WHITE, markeredgecolor=AXES_CLR, ms=8)

    k_dots_v = np.linspace(-2.8, 2.8, 21)
    E_dots_v = Ev - k_dots_v**2 / mv
    for x, y in zip(k_dots_v, E_dots_v):
        if y >= Efv:
            ax.plot(x, y, 'o', color=WHITE, markeredgecolor=AXES_CLR, ms=8) # holes are empty (white)
        else:
            ax.plot(x, y, 'o', color=AXES_CLR, ms=8) # filled states below Efv

    # Axes
    ax.axhline(0, color=AXES_CLR, lw=1) # k axis
    ax.axvline(0, color=AXES_CLR, lw=1) # E axis

    # Arrows for axes
    ax.annotate('', xy=(4.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=AXES_CLR, lw=1))
    ax.text(4.7, 0, 'k', va='center', ha='left', fontsize=16, style='italic')

    ax.annotate('', xy=(0, 7.8), xytext=(0, -4.8), arrowprops=dict(arrowstyle="->", color=AXES_CLR, lw=1))
    ax.text(0, 8.1, 'E', va='bottom', ha='center', fontsize=16, style='italic')

    # Lines and Labels for Energy Levels
    ax.plot([-3.5, 3.3], [Ec, Ec], color=AXES_CLR, ls='--', lw=1)
    ax.text(3.5, Ec, '$E_c$', va='center', fontsize=16)

    ax.plot([-3.6, 4.0], [Ev, Ev], color=AXES_CLR, ls='--', lw=1)
    ax.text(-3.8, Ev, '$E_v$', va='center', ha='right', fontsize=16)

    ax.plot([-2.0, 3.3], [Efc, Efc], color=AXES_CLR, ls='--', lw=1)
    ax.text(3.5, Efc, '$E_{FC}$', va='center', fontsize=16)

    ax.plot([-2.5, 3.3], [Efv, Efv], color=AXES_CLR, ls='--', lw=1)
    ax.text(3.5, Efv, '$E_{FV}$', va='center', fontsize=16)

    # Transition 'a' to 'b'
    k0 = 1.7
    Ea = Ec + k0**2 / mc
    Eb = Ev - k0**2 / mv

    ax.plot([0, 3.3], [Ea, Ea], color=AXES_CLR, ls='--', lw=1)
    ax.text(3.5, Ea, '$E_a$', va='center', fontsize=16)

    ax.plot([0, 3.3], [Eb, Eb], color=AXES_CLR, ls='--', lw=1)
    ax.text(3.5, Eb, '$E_b$', va='center', fontsize=16)

    # Vertical Arrow for emission
    ax.annotate('', xy=(k0, Eb), xytext=(k0, Ea), arrowprops=dict(arrowstyle="<->", color=CORAL, lw=2))
    ax.text(k0 + 0.2, (Ea+Eb)/2, r'$\hbar\omega(k)$', va='center', fontsize=16)

    # Points a and b
    ax.text(k0 + 0.35, Ea, 'a', va='center', ha='left', style='italic', fontsize=16)
    ax.text(k0 + 0.35, Eb, 'b', va='center', ha='left', style='italic', fontsize=16)

    # Eg Bracket
    ax.annotate('', xy=(-0.5, Ec), xytext=(-0.5, Ev), arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1))
    ax.text(-0.7, 0, '$E_g$', ha='right', va='center', fontsize=16)

    # Kinetic energy markers
    ax.annotate('', xy=(3.1, Ec), xytext=(3.1, Ea), arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1))
    ax.text(4.4, (Ea+Ec)/2, r'$\frac{\hbar^2 k^2}{2m_c} \equiv \hbar\omega_c$', ha='center', va='center', fontsize=16)

    ax.annotate('', xy=(3.1, Ev), xytext=(3.1, Eb), arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1))
    ax.text(4.4, (Eb+Ev)/2, r'$\frac{\hbar^2 k^2}{2m_v} \equiv \hbar\omega_v$', ha='center', va='center', fontsize=16)

    # Photon input
    x_wave = np.linspace(-4.0, -2.5, 100)
    y_wave = 0.5 + 0.2 * np.sin(25 * x_wave)
    ax.plot(x_wave, y_wave, color=TEAL, lw=2)
    ax.annotate('', xy=(-2.4, 0.5), xytext=(-2.5, 0.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(-3.25, 1.0, r'$I(\omega_0)$', ha='center', fontsize=16)

    # Limits
    ax.set_xlim(-4.5, 5.5)
    ax.set_ylim(-5, 8)

    # Remove axes lines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    save(fig, "band_gain_ek.jpg")


fig_12()


# ==============================================================================
#  FIG-13  Gain Formula Decomposition: ρ(ν) × [f_c − f_v] = γ₀(ν)
#  Three-panel conceptual figure showing the multiplication that produces gain.
#  Solid = T = 0 K (absolute zero);  Dashed = T > 0 K
# ==============================================================================
def fig_13():
    # Conceptual parameters for clear visualization
    Eg = 1.0
    Efc_Efv = 1.6  # quasi-Fermi separation (gain window edge)
    hnu = np.linspace(0, 2.5, 800)

    # ── Joint DOS: ρ(ν) ∝ sqrt(hν − Eg) ──────────────────────────────────────
    rho = np.sqrt(np.maximum(hnu - Eg, 0.0))
    rho_norm = rho / (np.max(rho) + 1e-12)

    # ── Inversion factor: f_c(ν) − f_v(ν) ────────────────────────────────────
    # For a conceptual plot, we assume symmetric bands (m_c = m_v). 
    # Mathematically, this exactly simplifies the Fermi-Dirac difference to a tanh function:
    # f_c(E_a) - f_v(E_b) = tanh((Efc_Efv - hnu) / (4 * kBT))
    
    # At T = 0 (solid): perfect step  +1 for hν < Efc−Efv,  −1 outside
    inv_T0 = np.where(hnu <= Efc_Efv, 1.0, -1.0)

    # At T > 0 (dashed): exact mathematical form for symmetric bands
    kBT_vis = 0.06  # pedagogical temperature to make smearing visible
    inv_T = np.tanh((Efc_Efv - hnu) / (4 * kBT_vis))

    # ── Product: γ₀(ν) = ρ(ν) × [f_c − f_v] ─────────────────────────────────
    gamma_T0 = rho_norm * inv_T0
    gamma_T  = rho_norm * inv_T

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # Apply consistent x-axis formatting to all three subplots
    for ax in axes:
        ax.set_xlim(0, 2.5)
        ax.set_xticks([0, Eg, Efc_Efv])
        ax.set_xticklabels([r"$0$", r"$E_g$", r"$E_{fc}\!-\!E_{fv}$"], fontsize=12)
        ax.axvline(Eg, color=AXES_CLR, lw=1.0, ls=":")
        ax.axvline(Efc_Efv, color=AXES_CLR, lw=1.0, ls=":")
        ax.set_xlabel(r"Photon energy $h\nu$", fontsize=13)
        ax.grid(True)

    # ── Panel (a): ρ(ν) ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(hnu, rho_norm, color=TEAL, lw=2.5)
    ax.text(Eg + 0.05, 0.85, r"$\rho(\nu) \propto \sqrt{h\nu - E_g}$",
            color=AXES_CLR, fontsize=11)
    ax.set_ylabel(r"$\rho(\nu)$", fontsize=13)
    ax.set_title(r"(a) Joint Density of States $\rho(\nu)$", pad=8)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0])
    ax.set_yticklabels(["$0$"], fontsize=12)

    # ── Panel (b): f_c − f_v ─────────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(0, color=AXES_CLR, lw=1.2)
    ax.plot(hnu, inv_T0, color=TEAL, lw=2.5, label=r"$T = 0$ K")
    ax.plot(hnu, inv_T,  color=TEAL, lw=2.5, ls="--", label=r"$T > 0$ K")
    ax.set_ylabel(r"$f_c - f_v$", fontsize=13)
    ax.set_title(r"(b) Population Inversion Factor", pad=8)
    ax.set_ylim(-1.4, 1.4)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["$-1$", "$0$", "$+1$"], fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    # ── Panel (c): γ₀(ν) = product ───────────────────────────────────────────
    ax = axes[2]
    ax.axhline(0, color=AXES_CLR, lw=1.5)
    ax.plot(hnu, gamma_T0, color=TEAL, lw=2.5, label=r"$T = 0$ K")
    ax.plot(hnu, gamma_T,  color=TEAL, lw=2.5, ls="--", label=r"$T > 0$ K")

    gain_peak_idx = np.argmax(gamma_T)
    ax.text(hnu[gain_peak_idx], gamma_T[gain_peak_idx] + 0.08,
            "Gain", color=TEAL, fontsize=12, ha="center", fontweight="bold")
    loss_region = hnu[hnu > Efc_Efv + 0.3]
    if len(loss_region):
        ax.text(loss_region[0], -0.25, "Loss", color=CORAL,
                fontsize=12, ha="center", fontweight="bold")

    ax.set_ylabel(r"$\gamma_0(\nu)$", fontsize=13)
    ax.set_title(r"(c) Net Gain $\gamma_0(\nu) = \rho(\nu) \cdot [f_c - f_v]$",
                 pad=8)
    ax.set_ylim(-0.8, 1.15)
    ax.set_yticks([0])
    ax.set_yticklabels(["$0$"], fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")

    fig.tight_layout(pad=2.0)
    save(fig, "gain_formula_decomposition.jpg")


fig_13()


# ==============================================================================
#  FIG-14  Gain Spectrum for Multiple Carrier Densities (Labelled)
#  Shows gain curves at increasing Δn with carrier density annotations,
#  gain/loss boundary, and bandwidth markers.
# ==============================================================================
def fig_14():
    Eg = 1.424       # GaAs bandgap eV
    kBT = 0.026      # room temperature
    me_eff = 0.067
    mh_eff = 0.50

    hnu = np.linspace(1.40, 1.56, 600)
    delta = np.maximum(hnu - Eg, 0.0)

    # Carrier densities and corresponding quasi-Fermi level positions
    carrier_configs = [
        (r"$\Delta n = 1.2 \times 10^{18}$", SKYBLUE, 0.01, -0.01),
        (r"$\Delta n = 1.4 \times 10^{18}$", LAVENDER, 0.04, 0.00),
        (r"$\Delta n = 1.6 \times 10^{18}$", TEAL,    0.06, 0.01),
        (r"$\Delta n = 1.8 \times 10^{18}$", CORAL,   0.08, 0.02),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Background shading
    ax.axhspan(0,    300, color=TEAL,  alpha=0.05)
    ax.axhspan(-160, 0,   color=CORAL, alpha=0.05)
    ax.axhline(0, color=AXES_CLR, lw=1.8)
    ax.axvline(Eg, color=AXES_CLR, lw=1.2, ls="--")
    ax.text(Eg + 0.001, -140, r"$E_g$", color=AXES_CLR, fontsize=11)

    # Compute and plot each gain curve
    gain_curves = {}
    unscaled = {}
    max_abs = 0.0

    for label, _, eta_c, eta_v in carrier_configs:
        frac_e = mh_eff / (me_eff + mh_eff)
        frac_h = me_eff / (me_eff + mh_eff)
        eps_e = frac_e * delta
        eps_h = frac_h * delta
        fc = 1.0 / (1.0 + np.exp((eps_e - eta_c) / kBT))
        fv_hole = 1.0 / (1.0 + np.exp((eps_h - eta_v) / kBT))
        inv = fc + fv_hole - 1.0
        g = np.sqrt(delta) * inv
        unscaled[label] = g
        max_abs = max(max_abs, np.max(np.abs(g)))

    G0 = 250.0 / (max_abs + 1e-12)
    for label, clr, _, _ in carrier_configs:
        gain = G0 * unscaled[label]
        gain_curves[label] = gain
        ax.plot(hnu, gain, color=clr, lw=2.5, label=label + r" cm$^{-3}$")

    # Gain bandwidth annotation on the strongest curve
    strongest_label = carrier_configs[-1][0]
    g_top = gain_curves[strongest_label]
    positive = g_top > 0
    if np.any(positive):
        idx_start = np.argmax(positive)
        idx_end   = np.where(positive)[0][-1]
        x1, x2 = hnu[idx_start], hnu[idx_end]
        ax.annotate("", xy=(x2, 30), xytext=(x1, 30),
                    arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
        ax.text((x1+x2)/2, 45,
                rf"$\sim{(x2-x1)*1000:.0f}$ meV gain BW",
                ha="center", color=AXES_CLR, fontsize=10)

    # Region labels
    ax.text(1.455, 230, "Gain", color=TEAL, fontsize=14,
            fontweight="bold", ha="center")
    ax.text(1.455, -130, "Loss", color=CORAL, fontsize=14,
            fontweight="bold", ha="center")

    ax.set_xlabel(r"Photon energy $h\nu$  (eV)", fontsize=13)
    ax.set_ylabel(r"Gain coefficient $\gamma$  (cm$^{-1}$)", fontsize=13)
    ax.set_title("Gain Spectrum vs. Carrier Density (GaAs, 300 K)", pad=10)
    ax.set_xlim(1.40, 1.56); ax.set_ylim(-160, 280)
    ax.legend(loc="upper right", framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "gain_spectrum_carrier_densities.jpg")


fig_14()


# ==============================================================================
#  FIG-15  Formation of the Heterojunction Band Diagram (3-Step Evolution)
#  Panel (a): Isolated materials — bands aligned to vacuum level
#  Panel (b): Thermal equilibrium — Fermi levels aligned, bands bend
#  Panel (c): Forward bias — quasi-Fermi levels split, carriers injected
# ==============================================================================
def fig_15():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    # Material parameters (conceptual, in eV and arbitrary position units)
    # P-cladding (wide gap)  |  Active (narrow gap)  |  N-cladding (wide gap)
    Eg_clad = 2.0
    Eg_act  = 1.4
    chi_clad = 4.0    # electron affinity of cladding
    chi_act  = 4.3    # electron affinity of active layer

    # Band discontinuities
    dEc = chi_act - chi_clad    # 0.3 eV (conduction band offset)
    dEv = (Eg_clad - Eg_act) - dEc  # 0.3 eV (valence band offset)

    # Spatial coordinates for each region
    x_p   = np.linspace(0, 2.0, 200)     # P-cladding
    x_act = np.linspace(2.0, 3.0, 100)   # Active layer
    x_n   = np.linspace(3.0, 5.0, 200)   # N-cladding

    # ── Panel (a): Isolated Materials ─────────────────────────────────────────
    ax = axes[0]

    # Vacuum level reference (flat)
    ax.hlines(0, -0.2, 5.2, color=AXES_CLR, lw=1.0, ls=":")
    ax.text(5.3, 0, r"$E_\mathrm{vac}$", va="center", fontsize=11, color=AXES_CLR)

    # P-cladding bands (isolated)
    Ec_p_iso = -chi_clad
    Ev_p_iso = Ec_p_iso - Eg_clad
    ax.hlines(Ec_p_iso, 0, 1.8, color=SKYBLUE, lw=2.5)
    ax.hlines(Ev_p_iso, 0, 1.8, color=CORAL, lw=2.5)
    Ef_p = Ev_p_iso + 0.15  # Fermi near valence (p-type)
    ax.hlines(Ef_p, 0, 1.8, color=AXES_CLR, lw=1.5, ls="--")

    # Active layer bands (isolated)
    Ec_act_iso = -chi_act
    Ev_act_iso = Ec_act_iso - Eg_act
    ax.hlines(Ec_act_iso, 2.2, 2.8, color=SKYBLUE, lw=2.5)
    ax.hlines(Ev_act_iso, 2.2, 2.8, color=CORAL, lw=2.5)
    Ef_act = (Ec_act_iso + Ev_act_iso) / 2  # intrinsic
    ax.hlines(Ef_act, 2.2, 2.8, color=AXES_CLR, lw=1.5, ls="--")

    # N-cladding bands (isolated)
    Ec_n_iso = -chi_clad
    Ev_n_iso = Ec_n_iso - Eg_clad
    ax.hlines(Ec_n_iso, 3.2, 5.0, color=SKYBLUE, lw=2.5)
    ax.hlines(Ev_n_iso, 3.2, 5.0, color=CORAL, lw=2.5)
    Ef_n = Ec_n_iso - 0.15  # Fermi near conduction (n-type)
    ax.hlines(Ef_n, 3.2, 5.0, color=AXES_CLR, lw=1.5, ls="--")

    # Electron affinity annotations
    ax.annotate("", xy=(0.3, Ec_p_iso), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.3))
    ax.text(0.1, (0 + Ec_p_iso)/2, r"$\chi_\mathrm{clad}$",
            color=GOLD, fontsize=10, ha="right", va="center")

    ax.annotate("", xy=(2.5, Ec_act_iso), xytext=(2.5, 0),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.3))
    ax.text(2.65, (0 + Ec_act_iso)/2, r"$\chi_\mathrm{act}$",
            color=GOLD, fontsize=10, ha="left", va="center")

    # Bandgap annotations
    ax.annotate("", xy=(1.5, Ev_p_iso), xytext=(1.5, Ec_p_iso),
                arrowprops=dict(arrowstyle="<->", color=LAVENDER, lw=1.3))
    ax.text(1.65, (Ec_p_iso + Ev_p_iso)/2, r"$E_{g,\mathrm{clad}}$",
            color=LAVENDER, fontsize=10, ha="left", va="center")

    ax.annotate("", xy=(2.35, Ev_act_iso), xytext=(2.35, Ec_act_iso),
                arrowprops=dict(arrowstyle="<->", color=LAVENDER, lw=1.3))
    ax.text(2.15, (Ec_act_iso + Ev_act_iso)/2, r"$E_{g,\mathrm{act}}$",
            color=LAVENDER, fontsize=10, ha="right", va="center")

    # Region labels
    ax.text(0.9, Ev_p_iso - 0.45, "P-clad", ha="center", color=CORAL, fontsize=10)
    ax.text(2.5, Ev_act_iso - 0.45, "Active", ha="center", color=TEAL, fontsize=10)
    ax.text(4.1, Ev_n_iso - 0.45, "N-clad", ha="center", color=SKYBLUE, fontsize=10)

    # Fermi level labels
    ax.text(-0.1, Ef_p, r"$E_{F,p}$", ha="right", va="center", fontsize=9, color=AXES_CLR)
    ax.text(5.1, Ef_n, r"$E_{F,n}$", ha="left", va="center", fontsize=9, color=AXES_CLR)

    # Band edge labels
    ax.text(-0.1, Ec_p_iso, r"$E_c$", ha="right", va="center", fontsize=10, color=SKYBLUE)
    ax.text(-0.1, Ev_p_iso, r"$E_v$", ha="right", va="center", fontsize=10, color=CORAL)

    # Separation lines
    for xsep in [2.0, 3.0]:
        ax.axvline(xsep, color=AXES_CLR, lw=1.0, ls=":", alpha=0.5)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(Ev_p_iso - 0.8, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"Electron energy $E$", fontsize=12)
    ax.set_title("(a) Isolated Materials", pad=10, fontsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Panel (b): Thermal Equilibrium (Zero Bias) ────────────────────────────
    ax = axes[1]

    # At equilibrium, the Fermi level is flat everywhere
    Ef_eq = -4.5  # chosen for visual clarity

    # Build smooth band bending using tanh transitions
    x_full = np.linspace(0, 5, 800)

    # Active layer boundaries
    x_left, x_right = 2.0, 3.0

    # Conduction band: clad level in bulk, dips down in active by dEc
    Ec_clad_eq = Ef_eq + 0.85  # position Ec_clad relative to Ef
    Ec_act_eq  = Ec_clad_eq - dEc  # active Ec is lower by dEc

    # Build Ec profile with smooth transitions at interfaces
    w_trans = 0.08  # transition width
    step_left  = 0.5 * (1 + np.tanh((x_full - x_left) / w_trans))
    step_right = 0.5 * (1 + np.tanh((x_full - x_right) / w_trans))
    # Goes: clad -> active -> clad
    Ec_eq = Ec_clad_eq - dEc * (step_left - step_right)

    # Valence band
    Ev_clad_eq = Ec_clad_eq - Eg_clad
    Ev_act_eq  = Ec_act_eq - Eg_act
    dEv_actual = Ev_act_eq - Ev_clad_eq  # positive means active Ev is higher
    Ev_eq = Ev_clad_eq + dEv_actual * (step_left - step_right)

    ax.plot(x_full, Ec_eq, color=SKYBLUE, lw=2.5)
    ax.plot(x_full, Ev_eq, color=CORAL, lw=2.5)
    ax.hlines(Ef_eq, 0, 5.0, color=AXES_CLR, lw=1.8, ls="--")

    # Band shading
    ax.fill_between(x_full, Ec_eq, Ec_eq + 0.35, color=SKYBLUE, alpha=0.08)
    ax.fill_between(x_full, Ev_eq - 0.35, Ev_eq, color=CORAL, alpha=0.08)

    # Active region shading
    ax.axvspan(x_left, x_right, color=TEAL, alpha=0.08)

    # dEc and dEv annotations
    ax.annotate("", xy=(x_right + 0.15, Ec_act_eq), xytext=(x_right + 0.15, Ec_clad_eq),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.3))
    ax.text(x_right + 0.25, (Ec_act_eq + Ec_clad_eq)/2, r"$\Delta E_c$",
            color=GOLD, fontsize=10, va="center")

    ax.annotate("", xy=(x_right + 0.15, Ev_act_eq), xytext=(x_right + 0.15, Ev_clad_eq),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.3))
    ax.text(x_right + 0.25, (Ev_act_eq + Ev_clad_eq)/2, r"$\Delta E_v$",
            color=GOLD, fontsize=10, va="center")

    # Labels
    ax.text(-0.1, Ef_eq, r"$E_F$", ha="right", va="center", fontsize=10, color=AXES_CLR)
    ax.text(-0.1, Ec_clad_eq, r"$E_c$", ha="right", va="center", fontsize=10, color=SKYBLUE)
    ax.text(-0.1, Ev_clad_eq, r"$E_v$", ha="right", va="center", fontsize=10, color=CORAL)

    # Interface boundaries
    for xd in [x_left, x_right]:
        ax.axvline(xd, color=AXES_CLR, lw=1.0, ls=":", alpha=0.5)

    # Region labels
    ax.text(1.0, Ev_clad_eq - 0.45, "P-clad", ha="center", color=CORAL, fontsize=10)
    ax.text(2.5, Ev_act_eq - 0.55, "Active", ha="center", color=TEAL, fontsize=10)
    ax.text(4.0, Ev_clad_eq - 0.45, "N-clad", ha="center", color=SKYBLUE, fontsize=10)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(Ev_clad_eq - 0.8, Ec_clad_eq + 0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) Thermal Equilibrium", pad=10, fontsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Panel (c): Forward Bias (Lasing Condition) ────────────────────────────
    ax = axes[2]

    # Under forward bias, quasi-Fermi levels split
    # The bands flatten and the well becomes more pronounced
    Efc = Ec_act_eq + 0.05   # quasi-Fermi for electrons (inside CB well)
    Efv = Ev_act_eq - 0.05   # quasi-Fermi for holes (inside VB well)

    # Under forward bias, the band slopes are reduced
    # P-side slopes down slightly, N-side slopes up slightly
    V_bias = 0.4
    slope_p = V_bias / (2 * x_left) if x_left > 0 else 0
    slope_n = -V_bias / (2 * (5.0 - x_right)) if (5.0 - x_right) > 0 else 0

    # Build biased profile
    Ec_bias = np.copy(Ec_eq)
    Ev_bias = np.copy(Ev_eq)

    # Add linear tilt to p and n regions
    for i, xv in enumerate(x_full):
        if xv < x_left:
            tilt = slope_p * (xv - x_left)
            Ec_bias[i] += tilt
            Ev_bias[i] += tilt
        elif xv > x_right:
            tilt = slope_n * (xv - x_right)
            Ec_bias[i] += tilt
            Ev_bias[i] += tilt

    ax.plot(x_full, Ec_bias, color=SKYBLUE, lw=2.5)
    ax.plot(x_full, Ev_bias, color=CORAL, lw=2.5)

    # Band shading
    ax.fill_between(x_full, Ec_bias, Ec_bias + 0.35, color=SKYBLUE, alpha=0.08)
    ax.fill_between(x_full, Ev_bias - 0.35, Ev_bias, color=CORAL, alpha=0.08)

    # Active region shading
    ax.axvspan(x_left, x_right, color=TEAL, alpha=0.08)

    # Quasi-Fermi levels (flat in the active region, extend into bulk)
    ax.hlines(Efc, 0.5, x_right + 0.3, color=SKYBLUE, lw=1.8, ls="--")
    ax.hlines(Efv, x_left - 0.3, 4.5, color=CORAL, lw=1.8, ls="--")

    # Labels
    ax.text(-0.1, Efc, r"$E_{Fc}$", ha="right", va="center", fontsize=10, color=SKYBLUE)
    ax.text(5.1, Efv, r"$E_{Fv}$", ha="left", va="center", fontsize=10, color=CORAL)

    # qV annotation
    ax.annotate("", xy=(4.5, Efv), xytext=(4.5, Efc),
                arrowprops=dict(arrowstyle="<->", color=LAVENDER, lw=1.4))
    ax.text(4.6, (Efc + Efv)/2, r"$qV$", color=LAVENDER, fontsize=11, va="center")

    # Electrons in the well (filled circles)
    for dx in [-0.25, -0.1, 0.0, 0.1, 0.25]:
        for dy in [0.04, 0.12, 0.20]:
            ax.plot(2.5 + dx, Ec_act_eq + dy, "o", color=SKYBLUE, ms=4, zorder=5)

    # Holes in the well (open circles)
    for dx in [-0.25, -0.1, 0.0, 0.1, 0.25]:
        for dy in [0.04, 0.12, 0.20]:
            ax.plot(2.5 + dx, Ev_act_eq - dy, "o", mfc="white", mec=CORAL,
                    ms=4, lw=1.0, zorder=5)

    # Photon emission arrow
    mid_E = (Ec_act_eq + Ev_act_eq) / 2
    ax.annotate("", xy=(2.5, Ev_act_eq + 0.05), xytext=(2.5, Ec_act_eq - 0.05),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.0,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(2.75, mid_E, r"$h\nu$", color=GOLD, fontsize=12, va="center")

    # Interface boundaries
    for xd in [x_left, x_right]:
        ax.axvline(xd, color=AXES_CLR, lw=1.0, ls=":", alpha=0.5)

    # Active layer width annotation
    ax.annotate("", xy=(x_left, Ev_clad_eq - 0.55), xytext=(x_right, Ev_clad_eq - 0.55),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.2))
    ax.text(2.5, Ev_clad_eq - 0.65, r"$d$", ha="center", color=AXES_CLR, fontsize=11)

    # Region labels
    ax.text(1.0, Ev_clad_eq - 0.45, "P-clad", ha="center", color=CORAL, fontsize=10)
    ax.text(2.5, Ev_act_eq - 0.55, "Active", ha="center", color=TEAL, fontsize=10)
    ax.text(4.0, Ev_clad_eq - 0.45, "N-clad", ha="center", color=SKYBLUE, fontsize=10)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(Ev_clad_eq - 0.8, Ec_clad_eq + 0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) Forward Bias (Lasing)", pad=10, fontsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=2.0)
    save(fig, "heterojunction_band_formation.jpg")


fig_15()


# ==============================================================================
#  FIG-16  DH Confinement: Stacked 3-Panel (Band Diagram + n_r + Optical Mode)
#  Unified plot showing how one structure solves both carrier and photon
#  confinement. All three panels share the same z-axis.
# ==============================================================================
def fig_16():
    z = np.linspace(-2.0, 2.0, 800)   # position in um
    d = 0.15                            # active layer half-width in um

    # ── (a) Energy Band Diagram ───────────────────────────────────────────────
    # Smooth transitions at z = ±d using tanh
    w_trans = 0.015  # transition width
    step_left  = 0.5 * (1 + np.tanh((z - (-d)) / w_trans))
    step_right = 0.5 * (1 + np.tanh((z - d) / w_trans))
    well = step_left - step_right  # 1 inside active, 0 outside

    Ec_clad = 1.7;  Ec_act = 1.4
    Ev_clad = 0.0;  Ev_act = 0.2
    Ec = Ec_clad - (Ec_clad - Ec_act) * well
    Ev = Ev_clad + (Ev_act - Ev_clad) * well

    # Quasi-Fermi levels
    Efc = 1.5;  Efv = 0.15

    # ── (b) Refractive Index Profile ──────────────────────────────────────────
    nr_clad = 3.40;  nr_act = 3.64
    nr = nr_clad + (nr_act - nr_clad) * well

    # ── (c) Optical Mode Profile ──────────────────────────────────────────────
    sigma_mode = 0.12
    mode = np.exp(-z**2 / (2 * sigma_mode**2))
    mode /= mode.max()

    # Confinement factor
    inside = np.abs(z) <= d
    G = np.trapezoid(mode[inside], z[inside]) / np.trapezoid(mode, z)

    # ── Create stacked figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 2]})

    # Common: active region shading and dashed boundary lines
    for ax in axes:
        ax.axvspan(-d, d, color=TEAL, alpha=0.06)
        ax.axvline(-d, color=AXES_CLR, lw=1.0, ls="--", alpha=0.6)
        ax.axvline(d,  color=AXES_CLR, lw=1.0, ls="--", alpha=0.6)

    # ── Panel (a): Band Diagram ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(z, Ec, color=SKYBLUE, lw=2.5, label=r"$E_c$")
    ax.plot(z, Ev, color=CORAL,   lw=2.5, label=r"$E_v$")

    # Band shading
    ax.fill_between(z, Ec, Ec + 0.25, color=SKYBLUE, alpha=0.08)
    ax.fill_between(z, Ev - 0.25, Ev, color=CORAL, alpha=0.08)

    # Quasi-Fermi levels
    ax.hlines(Efc, z[0], z[-1], color=SKYBLUE, lw=1.5, ls="--", label=r"$E_{Fc}$")
    ax.hlines(Efv, z[0], z[-1], color=CORAL,   lw=1.5, ls="--", label=r"$E_{Fv}$")

    # dEc and dEv annotations on the right interface
    ax.annotate("", xy=(d + 0.08, Ec_act), xytext=(d + 0.08, Ec_clad),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.3))
    ax.text(d + 0.15, (Ec_act + Ec_clad)/2, r"$\Delta E_c$",
            color=GOLD, fontsize=10, va="center")

    ax.annotate("", xy=(d + 0.08, Ev_act), xytext=(d + 0.08, Ev_clad),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.3))
    ax.text(d + 0.15, (Ev_act + Ev_clad)/2, r"$\Delta E_v$",
            color=GOLD, fontsize=10, va="center")

    # Electrons and holes in the well
    np.random.seed(42)
    for _ in range(12):
        xp = np.random.uniform(-d * 0.8, d * 0.8)
        yp = Ec_act + np.random.uniform(0.02, 0.18)
        ax.plot(xp, yp, "o", color=SKYBLUE, ms=4.5, zorder=5)
    for _ in range(12):
        xp = np.random.uniform(-d * 0.8, d * 0.8)
        yp = Ev_act - np.random.uniform(0.02, 0.14)
        ax.plot(xp, yp, "o", mfc="white", mec=CORAL, ms=4.5, lw=1.0, zorder=5)

    # Photon emission
    ax.annotate("", xy=(0, Ev_act + 0.03), xytext=(0, Ec_act - 0.03),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.0,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(0.06, (Ec_act + Ev_act)/2, r"$h\nu$", color=GOLD, fontsize=11, va="center")

    # Region labels
    ax.text(-1.2, Ec_clad + 0.15, r"P$^+$-cladding", ha="center", color=CORAL, fontsize=10)
    ax.text(0, Ec_act + 0.25, "Active", ha="center", color=TEAL, fontsize=10, fontweight="bold")
    ax.text(1.2, Ec_clad + 0.15, r"N$^+$-cladding", ha="center", color=SKYBLUE, fontsize=10)

    # Band edge labels
    ax.text(z[0] - 0.05, Ec_clad, r"$E_c$", ha="right", va="center", fontsize=10, color=SKYBLUE)
    ax.text(z[0] - 0.05, Ev_clad, r"$E_v$", ha="right", va="center", fontsize=10, color=CORAL)
    ax.text(z[-1] + 0.05, Efc, r"$E_{Fc}$", ha="left", va="center", fontsize=9, color=SKYBLUE)
    ax.text(z[-1] + 0.05, Efv, r"$E_{Fv}$", ha="left", va="center", fontsize=9, color=CORAL)

    ax.set_ylabel(r"Electron energy $E$", fontsize=12)
    ax.set_ylim(Ev_clad - 0.4, Ec_clad + 0.4)
    ax.set_yticks([])
    ax.set_title("(a) Band Diagram — Carrier Confinement", pad=8, fontsize=12)
    ax.grid(False)

    # ── Panel (b): Refractive Index ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(z, nr, color=TEAL, lw=2.5)
    ax.fill_between(z, nr_clad, nr, color=TEAL, alpha=0.12)

    # Delta n_r annotation
    ax.annotate("", xy=(d + 0.12, nr_act), xytext=(d + 0.12, nr_clad),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
    ax.text(d + 0.18, (nr_act + nr_clad)/2,
            rf"$\Delta n_r = {nr_act - nr_clad:.2f}$",
            color=AXES_CLR, fontsize=10, va="center")

    # Value labels
    ax.text(-1.2, nr_clad + 0.01, rf"$n_{{r,\mathrm{{clad}}}} = {nr_clad}$",
            ha="center", color=AXES_CLR, fontsize=9)
    ax.text(0, nr_act + 0.02, rf"$n_{{r,\mathrm{{act}}}} = {nr_act}$",
            ha="center", color=TEAL, fontsize=9)

    ax.set_ylabel(r"$n_r$", fontsize=13)
    ax.set_ylim(nr_clad - 0.08, nr_act + 0.08)
    ax.set_title("(b) Refractive Index — Waveguiding", pad=8, fontsize=12)
    ax.grid(False)

    # ── Panel (c): Optical Mode ───────────────────────────────────────────────
    ax = axes[2]
    ax.plot(z, mode, color=AXES_CLR, lw=2.5)
    ax.fill_between(z, 0, mode, where=inside, color=TEAL, alpha=0.20,
                    label=r"Gain region")
    outside = ~inside
    ax.fill_between(z, 0, mode, where=outside, color=CORAL, alpha=0.12,
                    label="Evanescent tail (loss)")

    # Gamma annotation
    ax.text(0, 0.5, rf"$\Gamma \approx {G*100:.0f}\%$",
            ha="center", color=TEAL, fontsize=13, fontweight="bold")

    ax.set_ylabel(r"$|\mathcal{E}(z)|^2$ (norm.)", fontsize=12)
    ax.set_xlabel(r"Position $z$", fontsize=13)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("(c) Optical Mode — Photon Confinement", pad=8, fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc", fontsize=10)
    ax.grid(False)

    # Active layer width annotation on bottom panel
    ax.annotate("", xy=(-d, -0.12), xytext=(d, -0.12),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.2),
                annotation_clip=False)
    ax.text(0, -0.18, r"$d$", ha="center", color=AXES_CLR, fontsize=12,
            clip_on=False)
    ax.set_ylim(-0.25, 1.15)

    # Custom x-ticks
    ax.set_xticks([-d, 0, d])
    ax.set_xticklabels([r"$-d/2$", "$0$", r"$d/2$"], fontsize=11)

    fig.tight_layout(h_pad=1.0)
    save(fig, "dh_confinement_stacked.jpg")


fig_16()
