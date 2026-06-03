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
#  FIG-5  Fermi Occupation Functions and Population Inversion
# ==============================================================================
def fig_05():
    kBT = 0.026
    configs = [
        (-0.03, -0.03, "--", "Low injection", 0.70),
        (0.08, 0.02, "-", "High injection", 1.00),
    ]
    delta = np.linspace(0.0, 0.18, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: f_c and 1-f_v for states linked by a direct transition.
    ax = axes[0]
    for (eta_c, eta_v, ls, lbl, alpha) in configs:
        _, fc, hole_occ = inversion_factor(delta, eta_c, eta_v, kBT)
        ax.plot(delta, fc, color=SKYBLUE, lw=2.5, ls=ls, alpha=alpha,
                label=rf"$f_c$  ({lbl})")
        ax.plot(delta, hole_occ, color=CORAL, lw=2.5, ls=ls, alpha=alpha,
                label=rf"$1-f_v$  ({lbl})")
    ax.axhline(0.5, color=AXES_CLR, lw=0.8, ls=":")
    ax.axhline(0,   color=AXES_CLR, lw=0.8)
    ax.axhline(1,   color=AXES_CLR, lw=0.8)
    ax.set_xlabel(r"Transition excess energy $h\nu - E_g$  (eV)", fontsize=12)
    ax.set_ylabel("Occupation probability", fontsize=12)
    ax.set_title(r"(a) Occupation of Coupled Electron-Hole States", pad=8)
    ax.set_xlim(0.0, 0.18); ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=9.5, framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    # Right: inversion factor fc - fv.
    ax = axes[1]
    ax.axhline(0, color=AXES_CLR, lw=1.5)
    diffs = {}
    for (eta_c, eta_v, ls, lbl, alpha) in configs:
        diff, _, _ = inversion_factor(delta, eta_c, eta_v, kBT)
        diffs[lbl] = diff
        ax.plot(delta, diff,
                color=TEAL if alpha > 0.9 else SKYBLUE,
                lw=2.5, ls=ls, alpha=alpha, label=lbl)

    diff_hi = diffs["High injection"]
    diff_lo = diffs["Low injection"]
    ax.fill_between(delta, 0, diff_hi, where=diff_hi >= 0, color=TEAL, alpha=0.14)
    ax.fill_between(delta, 0, diff_lo, where=diff_lo <= 0, color=CORAL, alpha=0.10)

    zero_hi = np.where(np.diff(np.sign(diff_hi)))[0]
    if len(zero_hi):
        x_zero = delta[zero_hi[0]]
        ax.annotate("", xy=(x_zero, 0.22), xytext=(0, 0.22),
                    arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.2))
        ax.text(x_zero/2, 0.27, f"~{x_zero*1000:.0f} meV gain window",
                ha="center", color=AXES_CLR, fontsize=10)

    ax.text(0.070, 0.42, "Gain", color=TEAL, fontsize=12, ha="center")
    ax.text(0.112, -0.40, "Absorption", color=CORAL, fontsize=11, ha="center")
    ax.set_xlabel(r"$h\nu - E_g$  (eV)", fontsize=12)
    ax.set_ylabel(r"$f_c - f_v$", fontsize=12)
    ax.set_title(r"(b) Population Inversion Factor for Direct Transitions", pad=8)
    ax.set_xlim(0.0, 0.18); ax.set_ylim(-0.95, 0.85)
    ax.legend(fontsize=10, framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout(pad=2.0)
    save(fig, "fermi_occupation_gain_window.jpg")


fig_05()


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
#  FIG-8  Optical Field Confinement: Homojunction vs. Double Heterojunction
# ==============================================================================
def fig_08():
    z = np.linspace(-3, 3, 600)   # position in um
    d_act = 0.2                   # active layer half-width in um

    # Refractive index profiles
    nr_homo = np.full_like(z, 3.64)
    nr_dh   = np.where(np.abs(z) <= d_act, 3.64, 3.50)

    # Optical mode: Gaussian (broad for homo, narrow for DH)
    sigma_homo = 1.20
    sigma_dh   = 0.15
    mode_homo  = np.exp(-z**2 / (2*sigma_homo**2))
    mode_dh    = np.exp(-z**2 / (2*sigma_dh**2))
    mode_homo /= mode_homo.max()
    mode_dh   /= mode_dh.max()

    # Confinement factors
    def Gamma(z, d, mode):
        inside = np.abs(z) <= d
        return np.trapezoid(mode[inside], z[inside]) / np.trapezoid(mode, z)
    G_homo = Gamma(z, d_act, mode_homo)
    G_dh   = Gamma(z, d_act, mode_dh)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    titles_top = ["Homojunction: Refractive Index",
                  "Double Heterojunction: Refractive Index"]
    titles_bot = ["Homojunction: Optical Mode",
                  "Double Heterojunction: Optical Mode"]
    nr_profiles = [nr_homo, nr_dh]
    modes       = [mode_homo, mode_dh]
    gammas      = [G_homo, G_dh]

    for col, (nr, mode, gamma, tt, tb) in enumerate(
            zip(nr_profiles, modes, gammas, titles_top, titles_bot)):
        # Top: refractive index
        ax = axes[0, col]
        ax.plot(z, nr, color=SKYBLUE, lw=2.5)
        ax.axvspan(-d_act, d_act, color=GOLD, alpha=0.15)
        for xd in [-d_act, d_act]:
            ax.axvline(xd, color=AXES_CLR, lw=0.9, ls="--")
        ax.set_ylabel(r"$n_r$", fontsize=12)
        ax.set_ylim(3.40, 3.75)
        ax.set_title(tt, pad=6, fontsize=11)
        ax.grid(True)
        if col == 1:
            ax.annotate("", xy=(d_act, 3.57), xytext=(d_act, 3.50),
                        arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.2))
            ax.text(d_act+0.1, 3.535, r"$\Delta n_r = 0.14$",
                    color=AXES_CLR, fontsize=9)
        else:
            ax.text(0, 3.66, "Constant $n_r$ — no waveguiding",
                    ha="center", color=CORAL, fontsize=9.5)

        # Bottom: optical mode
        ax = axes[1, col]
        ax.axvspan(-d_act, d_act, color=TEAL, alpha=0.20)
        outside = np.abs(z) > d_act
        ax.fill_between(z, 0, np.where(outside, mode, 0),
                        color=CORAL, alpha=0.15, label="Loss region")
        ax.plot(z, mode, color=AXES_CLR, lw=2.5)
        for xd in [-d_act, d_act]:
            ax.axvline(xd, color=AXES_CLR, lw=0.9, ls="--")
        gamma_y = 0.50 if col == 0 else 0.45
        gamma_x = 0 if col == 0 else 0.6
        ax.text(gamma_x, gamma_y,
                rf"$\Gamma \approx {gamma*100:.0f}\%$",
                ha="center", color=TEAL, fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$|\mathcal{E}|^2$  (norm.)", fontsize=12)
        ax.set_xlabel(r"Position $z$  ($\mu$m)", fontsize=12)
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(tb, pad=6, fontsize=11)
        ax.grid(True)

    fig.suptitle("Optical Confinement: Homojunction vs. Double Heterojunction",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "optical_confinement_comparison.jpg")


fig_08()

# ==============================================================================
#  FIG-11  Threshold Current: Homojunction vs. Double Heterojunction
# ==============================================================================
def fig_11():
    J = np.linspace(0, 50, 400)   # kA/cm^2

    # Modal gain: slope * J + intercept
    # Homojunction: small slope (poor confinement)
    slope_homo = 1.5; intcpt_homo = -20.0
    gamma_homo = slope_homo * J + intcpt_homo

    # DH: steep slope (high confinement)
    slope_dh = 30.0; intcpt_dh = -30.0
    gamma_dh = slope_dh * J + intcpt_dh

    alpha_th = 30.0   # threshold gain (cm^-1)

    # Threshold currents
    Jth_homo = (alpha_th - intcpt_homo) / slope_homo
    Jth_dh   = (alpha_th - intcpt_dh)   / slope_dh

    fig, ax = plt.subplots(figsize=(9, 6))

    # Gain/loss shading
    ax.axhspan(alpha_th, 110, color=TEAL,  alpha=0.08)
    ax.axhspan(-60, 0,        color=CORAL, alpha=0.06)
    ax.axhline(0,        color=AXES_CLR, lw=0.9)
    ax.plot([-0.5, 50], [alpha_th, alpha_th], color=AXES_CLR, lw=1.6, ls="--", zorder=4)
    ax.text(1, alpha_th, r"Loss threshold $\alpha_\mathrm{th}$",
            color=AXES_CLR, fontsize=10, va="center", bbox=dict(facecolor=WHITE, edgecolor='none', pad=2.0), zorder=10)

    ax.plot(J, gamma_homo, color=SKYBLUE, lw=2.5, ls="--",
            label="Homojunction")
    ax.plot(J, gamma_dh,   color=CORAL,   lw=2.5, ls="-",
            label="Double Heterojunction (DH)")

    # Vertical drop lines at thresholds
    threshold_labels = [
        (Jth_homo, SKYBLUE, r"$J_{th}^\mathrm{homo}$", "center", -55),
        (Jth_dh,   CORAL,   r"$J_{th}^\mathrm{DH}$",   "left",   -51),
    ]
    for Jth, clr, lbl, ha, ytxt in threshold_labels:
        if 0 <= Jth <= 50:
            ax.plot([Jth, Jth], [-48, 110], color=clr, lw=1.2, ls=":")
            xtext = Jth + (0.25 if ha == "left" else 0.0)
            ax.text(xtext, ytxt, lbl, ha=ha, color=clr, fontsize=10)

    # ~20x reduction annotation
    ax.annotate("", xy=(Jth_dh, alpha_th+18), xytext=(Jth_homo, alpha_th+18),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
    reduction = Jth_homo / Jth_dh
    ax.text((Jth_homo+Jth_dh)/2, alpha_th+22, rf"$\sim {reduction:.0f}\times$ reduction",
            ha="center", color=AXES_CLR, fontsize=10)

    ax.set_xlabel(r"Current density $J$  (kA$\cdot$cm$^{-2}$)", fontsize=13)
    ax.set_ylabel(r"Modal gain $\Gamma\gamma$  (cm$^{-1}$)", fontsize=13)
    ax.set_title("Threshold Current: Homojunction vs. Double Heterojunction", pad=10)
    ax.set_xlim(-0.5, 50); ax.set_ylim(-60, 110)
    ax.legend(loc="upper left", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "threshold_current_comparison.jpg")


fig_11()


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
    rho = np.where(hnu >= Eg, np.sqrt(hnu - Eg), 0.0)
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
        ax.set_xticklabels(["$0$", "$E_g$", "$E_{fc}\!-\!E_{fv}$"], fontsize=12)
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
