"""
Chapter 5 Plots — Semiconductor Physics Foundations
All figures follow the figure-generation-style guide.
Output: Chapter 5 - Plots/  (same directory as this script)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Shared palette ────────────────────────────────────────────────────────────
WHITE    = "#ffffff"
AXES_CLR = "#2b2b2b"
GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"
CORAL    = "#d94452"
GOLD     = "#c28800"
LAVENDER = "#7e57c2"
SKYBLUE  = "#1976d2"

# ── Shared rcParams ───────────────────────────────────────────────────────────
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


# ==============================================================================
#  FIG-L1-01  E-k Band Structure: Direct vs. Indirect Bandgap
# ==============================================================================
def fig_l1_01():
    k = np.linspace(-1, 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))

    # ── Panel (a): Direct (GaAs) ──────────────────────────────────────────────
    ax = axes[0]
    Ec, Ev = 1.5, 0.0
    ax.plot(k, Ec + 1.2*k**2, color=TEAL,  lw=2.5)
    ax.plot(k, Ev - 0.9*k**2, color=CORAL, lw=2.5)

    ax.annotate("", xy=(0, Ec), xytext=(0, Ev),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.0))
    ax.text(0.09, (Ec+Ev)/2, r"$h\nu = E_g$", color=GOLD, fontsize=11, va="center")

    ax.axhline(Ec, xmin=0.03, xmax=0.25, color=TEAL,  lw=1.2, ls="--")
    ax.axhline(Ev, xmin=0.03, xmax=0.25, color=CORAL, lw=1.2, ls="--")
    ax.text(-0.98, Ec+0.05, r"$E_c$", color=TEAL,  fontsize=11)
    ax.text(-0.98, Ev+0.05, r"$E_v$", color=CORAL, fontsize=11)

    ax.annotate("", xy=(-0.76, Ec), xytext=(-0.76, Ev),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.5))
    ax.text(-0.95, (Ec+Ev)/2, r"$E_g$", color=AXES_CLR, fontsize=11, ha="center")

    ax.text(0, 2.75, "Direct Bandgap (GaAs)", ha="center", fontsize=13,
            color=AXES_CLR, fontweight="bold")
    ax.text(0, 2.48, "Optical transition: Allowed", ha="center",
            fontsize=10.5, color=TEAL)
    ax.set_xlabel(r"Crystal momentum $k$"); ax.set_ylabel(r"Energy $E$  (a.u.)")
    ax.set_xticks([0]); ax.set_xticklabels([r"$0\;(\Gamma)$"])
    ax.set_yticks([]); ax.set_xlim(-1, 1); ax.set_ylim(-1.3, 3.0)
    ax.grid(True)

    # ── Panel (b): Indirect (Si) ──────────────────────────────────────────────
    ax = axes[1]
    k_cbm = 0.60
    CB_ind = Ec + 2.0*(k - k_cbm)**2
    VB_ind = Ev - 0.9*k**2
    ax.plot(k, CB_ind, color=TEAL,  lw=2.5)
    ax.plot(k, VB_ind, color=CORAL, lw=2.5)

    cbm_y = Ec
    ax.plot(k_cbm, cbm_y, "o", color=TEAL,  ms=7, zorder=5)
    ax.plot(0,     Ev,    "o", color=CORAL, ms=7, zorder=5)

    # Forbidden direct arrow (dashed)
    ax.annotate("", xy=(0, Ec), xytext=(0, Ev),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.8,
                                linestyle="dashed"))
    ax.text(0.07, (Ec+Ev)/2 + 0.15, "Not\nAllowed", color=CORAL, fontsize=9.5)

    # Phonon-assisted arrow
    ax.annotate("", xy=(k_cbm, cbm_y), xytext=(0, Ev),
                arrowprops=dict(arrowstyle="->", color=LAVENDER, lw=1.6,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(0.36, 0.88, r"$+$ phonon", color=LAVENDER, fontsize=9.5)

    ax.axhline(cbm_y, xmin=0.03, xmax=0.25, color=TEAL,  lw=1.2, ls="--")
    ax.axhline(Ev,    xmin=0.03, xmax=0.25, color=CORAL, lw=1.2, ls="--")
    ax.text(-0.98, cbm_y+0.05, r"$E_c$", color=TEAL,  fontsize=11)
    ax.text(-0.98, Ev   +0.05, r"$E_v$", color=CORAL, fontsize=11)
    ax.text(0,     -0.22, r"$k_{VBM}$", ha="center", color=CORAL, fontsize=10)
    ax.text(k_cbm, -0.22, r"$k_{CBM}$", ha="center", color=TEAL,  fontsize=10)

    ax.annotate("", xy=(-0.76, cbm_y), xytext=(-0.76, Ev),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.5))
    ax.text(-0.95, (cbm_y+Ev)/2, r"$E_g$", color=AXES_CLR, fontsize=11, ha="center")

    ax.text(0, 2.75, "Indirect Bandgap (Si)", ha="center", fontsize=13,
            color=AXES_CLR, fontweight="bold")
    ax.text(0, 2.48, "Direct transition: Forbidden", ha="center",
            fontsize=10.5, color=CORAL)
    ax.set_xlabel(r"Crystal momentum $k$")
    ax.set_xticks([0, k_cbm])
    ax.set_xticklabels([r"$\Gamma$", r"$k_{CBM}$"])
    ax.set_yticks([]); ax.set_xlim(-1, 1); ax.set_ylim(-1.3, 3.0)
    ax.grid(True)

    fig.tight_layout(pad=2.0)
    save(fig, "ek_band_structure.jpg")


fig_l1_01()


# ==============================================================================
#  FIG-L1-02  Density of States and Carrier Distribution (4-Panel)
# ==============================================================================
def fig_l1_02():
    Ec, Ev, Ef = 1.5, 0.0, 0.75
    kBT = 0.15  # enlarged for pedagogical visibility (true kBT~0.026)
    E_cb  = np.linspace(Ec, Ec+1.2, 300)
    E_vb  = np.linspace(Ev-1.2, Ev, 300)
    E_all = np.linspace(Ev-1.4, Ec+1.4, 600)
    g_cb  = np.sqrt(np.maximum(E_cb-Ec, 0))
    g_vb  = np.sqrt(np.maximum(Ev-E_vb, 0))
    f_all = 1.0/(1.0+np.exp((E_all-Ef)/kBT))
    n_E   = g_cb*(1.0/(1.0+np.exp((E_cb-Ef)/kBT)))
    p_E   = g_vb*(1.0-1.0/(1.0+np.exp((E_vb-Ef)/kBT)))

    fig, axes = plt.subplots(1, 4, figsize=(14, 6.0), sharey=True)

    # (a) Band diagram
    ax = axes[0]
    ax.fill_betweenx([Ec, Ec+1.2], 0, 0.8, color=TEAL,  alpha=0.18)
    ax.fill_betweenx([Ev-1.2, Ev], 0, 0.8, color=CORAL, alpha=0.18)
    ax.axhline(Ec, color=TEAL,  lw=2.0, ls="--")
    ax.axhline(Ev, color=CORAL, lw=2.0, ls="--")
    ax.axhline(Ef, color=GOLD,  lw=1.8)
    for dy in [0.10, 0.22, 0.36]:
        ax.plot(0.40, Ec+dy, "o", color=TEAL, ms=7, zorder=5)
        ax.plot(0.40, Ev-dy, "o", mfc="white", mec=CORAL, ms=7, lw=1.5, zorder=5)
    ax.text(0.83, Ec+0.04, r"$E_c$", color=TEAL,  fontsize=11)
    ax.text(0.83, Ev+0.04, r"$E_v$", color=CORAL, fontsize=11)
    ax.text(0.83, Ef+0.04, r"$E_F$", color=GOLD,  fontsize=11)
    ax.set_xlim(0, 1.0); ax.set_ylim(-1.4, 2.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) Band Diagram", pad=8)
    ax.set_ylabel(r"Energy $E$  (a.u.)", fontsize=12)
    ax.grid(False)

    # (b) g(E)
    ax = axes[1]
    ax.plot(g_cb, E_cb, color=TEAL,  lw=2.5)
    ax.plot(g_vb, E_vb, color=CORAL, lw=2.5)
    ax.axhline(Ec, color=TEAL,  lw=1.2, ls="--")
    ax.axhline(Ev, color=CORAL, lw=1.2, ls="--")
    ax.text(0.05, Ec+0.70, r"$g_c(E)\propto\sqrt{E-E_c}$", color=TEAL,  fontsize=9.5)
    ax.text(0.05, Ev-0.70, r"$g_v(E)\propto\sqrt{E_v-E}$", color=CORAL, fontsize=9.5)
    ax.text(0.88, Ec+0.04, r"$E_c$", color=TEAL,  fontsize=11)
    ax.text(0.88, Ev+0.04, r"$E_v$", color=CORAL, fontsize=11)
    ax.set_xlim(-0.05, 1.15); ax.set_xticks([])
    ax.set_title(r"(b) Density of States $g(E)$", pad=8)
    ax.grid(True)

    # (c) f(E) and 1-f(E)
    ax = axes[2]
    ax.plot(f_all,   E_all, color=TEAL,  lw=2.5, label=r"$f(E)$")
    ax.plot(1-f_all, E_all, color=CORAL, lw=2.5, ls="--", label=r"$1-f(E)$")
    ax.axhline(Ef, color=GOLD, lw=1.5)
    ax.text(0.55, Ef+0.12, r"$E_F$", color=GOLD, fontsize=11)
    ax.axvline(0.5, color=AXES_CLR, lw=0.8, ls=":")
    ax.legend(loc="upper right", framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.set_xlim(-0.05, 1.05); ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_xlabel("Occupation probability", fontsize=11)
    ax.set_title(r"(c) Fermi-Dirac $f(E)$", pad=8)
    ax.grid(True)

    # (d) n_E and p_E
    ax = axes[3]
    ax.fill_betweenx(E_cb, 0, n_E, color=TEAL,  alpha=0.35)
    ax.plot(n_E, E_cb, color=TEAL,  lw=2.5)
    ax.fill_betweenx(E_vb, 0, p_E, color=CORAL, alpha=0.35)
    ax.plot(p_E, E_vb, color=CORAL, lw=2.5)
    ax.axhline(Ec, color=TEAL,  lw=1.2, ls="--")
    ax.axhline(Ev, color=CORAL, lw=1.2, ls="--")
    mx = max(float(np.max(n_E)), float(np.max(p_E)))
    # Labels in white space with arrows into the shaded fill
    pk_n = int(np.argmax(n_E))
    pk_p = int(np.argmax(p_E))
    ax.annotate(r"Area $= n$",
                xy=(float(n_E[pk_n])*0.45, float(E_cb[pk_n])),
                xytext=(mx*1.35, Ec+0.55),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2,
                                connectionstyle="arc3,rad=-0.15"),
                color=TEAL, fontsize=10, ha="center")
    ax.annotate(r"Area $= p$",
                xy=(float(p_E[pk_p])*0.45, float(E_vb[pk_p])),
                xytext=(mx*1.35, Ev-0.55),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2,
                                connectionstyle="arc3,rad=0.15"),
                color=CORAL, fontsize=10, ha="center")
    ax.set_xlim(0, mx*1.8); ax.set_xticks([])
    ax.set_title(r"(d) Carrier Distributions $n_E,\,p_E$", pad=8)
    ax.grid(True)

    fig.tight_layout(pad=1.5)
    save(fig, "dos_carrier_distribution.jpg")



fig_l1_02()


# ==============================================================================
#  FIG-L1-03  Quasi-Fermi Levels Under Carrier Injection
# ==============================================================================
def fig_l1_03():
    Ec, Ev   = 2.2, 0.0
    Efn, Efp = 2.65, -0.45

    fig, ax = plt.subplots(figsize=(6.5, 7.0))

    ax.fill_betweenx([Ec, Ec+1.0], -0.1, 1.0, color=TEAL,  alpha=0.12)
    ax.fill_betweenx([Ev-1.0, Ev], -0.1, 1.0, color=CORAL, alpha=0.12)
    ax.axhline(Ec,  color=TEAL,    lw=2.0)
    ax.axhline(Ev,  color=CORAL,   lw=2.0)
    ax.axhline(Efn, color=TEAL,    lw=1.8, ls="--")
    ax.axhline(Efp, color=CORAL,   lw=1.8, ls="--")

    ax.text(1.08, Ec +0.03, r"$E_c$",    color=TEAL,    fontsize=12, va="bottom")
    ax.text(1.08, Ev +0.03, r"$E_v$",    color=CORAL,   fontsize=12, va="bottom")
    ax.text(1.08, Efn+0.03, r"$E_{fn}$", color=TEAL,    fontsize=12, va="bottom")
    ax.text(1.08, Efp+0.03, r"$E_{fp}$", color=CORAL,   fontsize=12, va="bottom")

    for i, dx in enumerate([0.20, 0.35, 0.50]):
        ax.plot(dx, Ec+0.10+0.16*i, "o", color=TEAL,  ms=8, zorder=5)
        ax.plot(dx, Ev-0.10-0.16*i, "o", mfc="white", mec=CORAL, ms=8, lw=1.5, zorder=5)

    # E_g brace
    ax.annotate("", xy=(-0.07, Ec), xytext=(-0.07, Ev),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.5))
    ax.text(-0.09, (Ec+Ev)/2, r"$E_g$", color=AXES_CLR, fontsize=11,
            ha="right", va="center")

    # Efn-Efp brace
    ax.annotate("", xy=(1.03, Efn), xytext=(1.03, Efp),
                arrowprops=dict(arrowstyle="<->", color=LAVENDER, lw=1.5))
    ax.text(1.05, (Efn+Efp)/2, r"$E_{fn}-E_{fp}$",
            color=LAVENDER, fontsize=10.5, ha="left", va="center")

    # Photon transition arrow (Emission)
    E_cb_st, E_vb_st = Ec+0.30, Ev-0.30
    ax.annotate("", xy=(0.75, E_vb_st), xytext=(0.75, E_cb_st),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.2))
    ax.text(0.80, (E_cb_st+E_vb_st)/2, r"$h\nu$",
            color=GOLD, fontsize=12, va="center")

    # Gain window shading
    ax.axhspan(Efp, Efn, color=TEAL, alpha=0.05)
    ax.text(0.02, (Efn+Ec)/2, "Gain window\n"r"$E_g < h\nu < E_{fn}-E_{fp}$",
            color=TEAL, fontsize=9.5, va="center")

    ax.set_xlim(-0.15, 1.45); ax.set_ylim(-1.5, 3.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(r"Energy $E$  (a.u.)", fontsize=12)
    ax.set_title("Quasi-Fermi Levels Under Carrier Injection", fontsize=15, pad=10)
    ax.grid(False)

    fig.tight_layout()
    save(fig, "quasi_fermi_levels.jpg")


fig_l1_03()


# ==============================================================================
#  FIG-L1-04  Population Inversion: Single Material vs. PN Junction
# ==============================================================================
def fig_l1_04():
    kBT = 0.026
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))

    # ── (a) Single material ───────────────────────────────────────────────────
    ax = axes[0]
    Ec, Ev, Ef = 2.0, 0.0, 1.0
    E2, E1     = Ec+0.35, Ev-0.35

    ax.fill_betweenx([Ec, Ec+0.9], -0.1, 1.05, color=TEAL,  alpha=0.13)
    ax.fill_betweenx([Ev-0.9, Ev], -0.1, 1.05, color=CORAL, alpha=0.13)
    
    ax.plot([-0.1, 1.05], [Ec, Ec], color=TEAL,  lw=2.0)
    ax.plot([-0.1, 1.05], [Ev, Ev], color=CORAL, lw=2.0)
    ax.plot([-0.1, 1.05], [Ef, Ef], color=GOLD,  lw=1.8, ls="--")
    ax.plot([0.3, 1.05], [E2, E2], color=TEAL,  lw=1.5, ls=":")
    ax.plot([0.3, 1.05], [E1, E1], color=CORAL, lw=1.5, ls=":")

    ax.text(1.08, E2, r"$E_2$ (CB)", color=TEAL,  fontsize=10.5, va="center")
    ax.text(1.08, E1, r"$E_1$ (VB)", color=CORAL, fontsize=10.5, va="center")
    ax.text(1.08, Ef, r"$E_F$",      color=GOLD,  fontsize=10.5, va="center")
    ax.text(1.08, Ec, r"$E_c$",      color=TEAL,  fontsize=10.5, va="center")
    ax.text(1.08, Ev, r"$E_v$",      color=CORAL, fontsize=10.5, va="center")

    f2 = 1/(1+np.exp((E2-Ef)/kBT))
    f1 = 1/(1+np.exp((E1-Ef)/kBT))
    ax.text(0.05, 3.05, rf"$f(E_2) = {f2:.2e}$",     color=TEAL,  fontsize=10)
    ax.text(0.05, 2.90, rf"$f(E_1) = {f1:.4f}$",     color=CORAL, fontsize=10)
    ax.text(0.05, 2.75, r"$f(E_2)\ll f(E_1)$: No inversion!", color=CORAL, fontsize=10, fontstyle="italic")
    ax.text(0.5, (E2+E1)/2, r"$\times$", color=CORAL, fontsize=22, ha="center", va="center")
    ax.text(0.5, -1.30, "Not Possible — Single Material",
            ha="center", color=CORAL, fontsize=11, fontweight="bold")

    ax.set_xlim(-0.15, 1.50); ax.set_ylim(-1.7, 3.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(r"Energy $E$  (a.u.)", fontsize=12)
    ax.set_title("(a) Single Uniform Material", pad=8)
    ax.grid(False)

    # ── (b) PN junction ───────────────────────────────────────────────────────
    ax = axes[1]
    x = np.linspace(0, 1, 400)
    def bend(xarr, hi, lo, x0=0.38, x1=0.62, s=3.0):
        m = (x0+x1)/2; h = (x1-x0)/2
        return hi + (lo-hi)*0.5*(1+np.tanh((xarr-m)/h*s))

    Ec_pb, Ec_nb = 2.0, 1.6
    Ev_pb, Ev_nb = 0.0, -0.4
    Efn_b, Efp_b = Ec_nb+0.3, Ev_pb-0.3

    ax.plot(x, bend(x, Ec_pb, Ec_nb), color=TEAL,  lw=2.5)
    ax.plot(x, bend(x, Ev_pb, Ev_nb), color=CORAL, lw=2.5)
    ax.fill_between(x, bend(x,Ec_pb,Ec_nb), bend(x,Ec_pb,Ec_nb)+0.55, color=TEAL,  alpha=0.10)
    ax.fill_between(x, bend(x,Ev_pb,Ev_nb)-0.55, bend(x,Ev_pb,Ev_nb), color=CORAL, alpha=0.10)
    ax.plot([-0.05, 1.0], [Efn_b, Efn_b], color=TEAL,  lw=1.8, ls="--")
    ax.plot([-0.05, 1.0], [Efp_b, Efp_b], color=CORAL, lw=1.8, ls="--")
    ax.axvspan(0.38, 0.62, color=LAVENDER, alpha=0.08)

    ax.text(1.02, Efn_b, r"$E_{fn}$", color=TEAL,  fontsize=11, va="center")
    ax.text(1.02, Efp_b, r"$E_{fp}$", color=CORAL, fontsize=11, va="center")
    ax.text(0.5, -1.40, "Depletion Region", ha="center", color=LAVENDER, fontsize=9.5)

    # Gain window brace
    ax.annotate("", xy=(0.18, Efn_b), xytext=(0.18, Efp_b),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.5))
    ax.text(0.20, (Efn_b+Efp_b)/2, r"$E_{fn}-E_{fp}>E_g$",
            color=GOLD, fontsize=10, va="center")

    # Emission arrow in depletion
    Ec_mid = bend(np.array([0.5]), Ec_pb, Ec_nb)[0]+0.12
    Ev_mid = bend(np.array([0.5]), Ev_pb, Ev_nb)[0]-0.12
    ax.annotate("", xy=(0.5, Ev_mid), xytext=(0.5, Ec_mid),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.0))
    ax.text(0.54, (Ec_mid+Ev_mid)/2, r"$h\nu$", color=GOLD, fontsize=11)

    ax.text(0.15, -1.05, "p-side", ha="center", color=CORAL, fontsize=10)
    ax.text(0.85, -1.05, r"n-side", ha="center", color=TEAL,  fontsize=10)
    ax.text(0.5, -1.30, "Allowed — Forward-Biased PN Junction",
            ha="center", color=TEAL, fontsize=11, fontweight="bold")

    ax.set_xlim(-0.05, 1.30); ax.set_ylim(-1.7, 2.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Forward-Biased PN Junction", pad=8)
    ax.grid(False)

    fig.tight_layout(pad=2.0)
    save(fig, "population_inversion_condition.jpg")


fig_l1_04()


# ==============================================================================
#  FIG-L1-05  Homojunction Band Diagram: Three Bias States
# ==============================================================================
def fig_l1_05():
    x = np.linspace(0, 1, 500)

    def bend(xarr, hi, lo, x0=0.38, x1=0.62, s=3.0):
        m = (x0+x1)/2; h = (x1-x0)/2
        return hi + (lo-hi)*0.5*(1+np.tanh((xarr-m)/h*s))

    V0   = 1.1
    Ec_p = 2.2;  Ec_n = Ec_p - V0
    Ev_p = 0.0;  Ev_n = Ev_p - V0
    Ef_eq = (Ec_n + Ev_p) / 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.0), sharey=True)

    def draw(ax, Ec_x, Ev_x, ef_lines, annots, title, ylim=(-2.0, 3.2)):
        ax.plot(x, Ec_x, color=TEAL,  lw=2.5)
        ax.plot(x, Ev_x, color=CORAL, lw=2.5)
        ax.fill_between(x, Ec_x, Ec_x+0.6, color=TEAL,  alpha=0.10)
        ax.fill_between(x, Ev_x-0.6, Ev_x,  color=CORAL, alpha=0.10)
        ax.axvspan(0.38, 0.62, color=LAVENDER, alpha=0.07)
        for (yv, clr, ls, lbl) in ef_lines:
            ax.plot([-0.05, 0.98], [yv, yv], color=clr, lw=1.7, ls=ls)
            ax.text(1.02, yv, lbl, color=clr, fontsize=10, va="center")
        for (xp, yp, txt, clr, fs) in annots:
            ax.text(xp, yp, txt, color=clr, fontsize=fs, ha="center", va="center")
        ax.text(0.18, ylim[0]+0.15, "p-side",     ha="center", color=CORAL, fontsize=9.5)
        ax.text(0.82, ylim[0]+0.15, r"n$^+$-side", ha="center", color=TEAL,  fontsize=9.5)
        ax.set_xlim(-0.05, 1.30); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, pad=8); ax.grid(False)

    # (a) Equilibrium
    Ec_eq = bend(x, Ec_p, Ec_n); Ev_eq = bend(x, Ev_p, Ev_n)
    axes[0].plot([0.3, 1.0], [Ec_p, Ec_p], color=TEAL, lw=1.0, ls=":")
    axes[0].annotate("", xy=(0.97, Ec_n), xytext=(0.97, Ec_p),
                     arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
    axes[0].text(1.12, (Ec_p+Ec_n)/2, r"$eV_0$", color=AXES_CLR, fontsize=10, va="center")
    for dy in [0.10, 0.22, 0.36]:
        axes[0].plot(0.78+dy*0.3, Ec_n+dy, "o", color=TEAL, ms=6, zorder=5)
        axes[0].plot(0.22-dy*0.3, Ev_p-dy, "o", mfc="white", mec=CORAL, ms=6, lw=1.4, zorder=5)
    draw(axes[0], Ec_eq, Ev_eq,
         ef_lines=[(Ef_eq, GOLD, "-", r"$E_F$")],
         annots=[], title=r"(a) Equilibrium  $(V=0)$")
    axes[0].set_ylabel(r"Energy $E$  (a.u.)", fontsize=12)

    # (b) Forward bias
    V_f = 0.55; Ec_nb = Ec_n+V_f; Ev_nb = Ev_n+V_f
    Ec_fwd = bend(x, Ec_p, Ec_nb); Ev_fwd = bend(x, Ev_p, Ev_nb)
    Efn_f = Ec_nb+0.10; Efp_f = Ev_p-0.10
    axes[1].plot([0.3, 1.0], [Ec_p, Ec_p], color=TEAL, lw=1.0, ls=":")
    axes[1].annotate("", xy=(0.97, Ec_nb), xytext=(0.97, Ec_p),
                     arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
    axes[1].text(1.12, (Ec_p+Ec_nb)/2, r"$e(V_0-V)$", color=AXES_CLR, fontsize=9, va="center")
    axes[1].annotate("", xy=(0.15, Efn_f), xytext=(0.15, Efp_f),
                     arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.3))
    axes[1].text(0.18, (Efn_f+Efp_f)/2, r"$eV$", color=GOLD, fontsize=10, va="center")
    Ec_m = bend(np.array([0.5]),Ec_p,Ec_nb)[0]+0.10
    Ev_m = bend(np.array([0.5]),Ev_p,Ev_nb)[0]-0.10
    axes[1].annotate("", xy=(0.5, Ev_m), xytext=(0.5, Ec_m),
                     arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.0))
    axes[1].text(0.55, (Ec_m+Ev_m)/2, r"$h\nu\approx E_g$", color=GOLD, fontsize=9.5, va="center")
    draw(axes[1], Ec_fwd, Ev_fwd,
         ef_lines=[(Efn_f,TEAL,"--",r"$E_{fn}$"),(Efp_f,CORAL,"--",r"$E_{fp}$")],
         annots=[], title=r"(b) Forward Bias  $(V>0)$")

    # (c) High forward bias — gain condition
    V_g = 0.90; Ec_ng = Ec_n+V_g; Ev_ng = Ev_n+V_g
    Ec_gain = bend(x, Ec_p, Ec_ng); Ev_gain = bend(x, Ev_p, Ev_ng)
    Efn_g = Ec_ng+0.38; Efp_g = Ev_p-0.38
    axes[2].axhspan(Efp_g, Efn_g, color=GOLD, alpha=0.06)
    axes[2].annotate("", xy=(0.13, Efn_g), xytext=(0.13, Efp_g),
                     arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.5))
    axes[2].text(0.15, (Efn_g+Efp_g)/2,
                 r"$E_{fn}-E_{fp}>E_g$" "\nGain window",
                 color=GOLD, fontsize=9, va="center")
    draw(axes[2], Ec_gain, Ev_gain,
         ef_lines=[(Efn_g,TEAL,"--",r"$E_{fn}$"),(Efp_g,CORAL,"--",r"$E_{fp}$")],
         annots=[], title="(c) High Forward Bias — Gain Condition")

    fig.tight_layout(pad=1.5)
    save(fig, "homojunction_bias_states.jpg")


fig_l1_05()
