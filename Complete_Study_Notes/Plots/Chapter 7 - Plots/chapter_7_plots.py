"""
Chapter 7 Plots — Heterojunctions, Gain, and LEDs
All Python figures follow the figure-generation-style guide.
Figures 3 (binary connectivity) and 5 (recombination mechanisms) are AI-generated
— see Chapter 7 Plots.md for prompts.
Output: Chapter 7 - Plots/ (same directory as this script)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
#  FIG-1  Kramers-Kronig: Real and Imaginary Susceptibility
# ==============================================================================
def fig_01():
    w = np.linspace(-3, 3, 800)
    w0 = 0.0   # resonance

    # chi2: Lorentzian absorption peak (imaginary part)
    gamma = 0.5
    chi2 = 45 * gamma**2 / ((w - w0)**2 + gamma**2)

    # chi1: dispersive (real part) — derivative-like S-curve via KK
    chi1 = 25 * (-(w - w0)) * gamma / ((w - w0)**2 + gamma**2) * 2

    # Kernel peak: sharp 1/(w'-w) spike at resonance (capped for display)
    kernel = 60 * np.exp(-((w - w0)**2) / 0.02)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor(WHITE)

    # Kernel (yellow spike)
    ax.fill_between(w, 0, kernel, color=GOLD, alpha=0.35, label=r"Kernel $1/(\omega'-\omega)$")
    ax.plot(w, kernel, color=GOLD, lw=1.8)

    # Green shading under kernel to left of resonance
    mask_left = w < w0
    ax.fill_between(w[mask_left], 0, kernel[mask_left], color=MINT, alpha=0.30)

    # chi2 (pink/salmon bell)
    ax.fill_between(w, 0, chi2, color=PINK, alpha=0.20)
    ax.plot(w, chi2, color=PINK, lw=2.5, label=r"$\chi_2(\omega')$")

    # chi1 (blue S-curve)
    ax.plot(w, chi1, color=SKYBLUE, lw=2.5, label=r"$\chi_1(\omega)$")
    ax.plot(w0, 0, "o", color=SKYBLUE, ms=8, zorder=6)  # evaluation point

    # Resonance line
    ax.axvline(w0, color=AXES_CLR, lw=1.5)
    ax.text(w0 + 0.05, 56, r"$\omega'$", color=AXES_CLR, fontsize=12)

    # Direct labels on curves
    ax.text(-1.8, 22, r"$\chi_1(\omega)$", color=SKYBLUE, fontsize=11)
    ax.text( 1.0, 40, r"$\chi_2(\omega')$", color=PINK,   fontsize=11)
    ax.text( 0.08, 52, r"$\sim 1/(\omega'-\omega)$", color=GOLD, fontsize=9.5)

    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.set_xlabel(r"Angular frequency $\omega$", fontsize=13)
    ax.set_yticks([])
    ax.set_xlim(-3, 3); ax.set_ylim(-35, 68)
    ax.set_title("Kramers-Kronig Relation: Dispersion and Absorption", pad=10)
    ax.legend(loc="upper right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(False)

    fig.tight_layout()
    save(fig, "kramers_kronig.jpg")


fig_01()


# ==============================================================================
#  FIG-2  III-V Semiconductor Periodic Table Section
# ==============================================================================
def fig_02():
    groups = {
        "II":  [("Zn", "Zinc"), ("Cd", "Cadmium"), ("Hg", "Mercury")],
        "III": [("Al", "Aluminum"), ("Ga", "Gallium"), ("In", "Indium")],
        "IV":  [("Si", "Silicon"), ("Ge", "Germanium"), (None, None)],
        "V":   [("P", "Phosphorus"), ("As", "Arsenic"), ("Sb", "Antimony")],
        "VI":  [("S", "Sulfur"), ("Se", "Selenium"), ("Te", "Tellurium")],
    }

    col_order = ["II", "III", "IV", "V", "VI"]
    n_cols = len(col_order); n_rows = 3

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows + 0.8)
    ax.axis("off")

    cell_w, cell_h = 1.0, 1.0
    pad = 0.05

    for ci, grp in enumerate(col_order):
        # Column header
        ax.text(ci + 0.5, n_rows + 0.4, f"Group {grp}",
                ha="center", va="center", fontsize=11,
                fontweight="bold", color=AXES_CLR)

        for ri, (sym, name) in enumerate(groups[grp]):
            y = n_rows - ri - 1
            x = ci

            if sym is None:
                # Empty cell placeholder
                rect = mpatches.FancyBboxPatch(
                    (x + pad, y + pad), cell_w - 2*pad, cell_h - 2*pad,
                    boxstyle="round,pad=0.02",
                    facecolor="#f0f0f0", edgecolor="#cccccc", lw=1.0)
                ax.add_patch(rect)
                continue

            # Highlight Group III and V (the compound-forming groups)
            is_active = grp in ["III", "V"]
            fc = TEAL + "22" if is_active else "#f0f0f0"
            ec = TEAL if is_active else "#aaaaaa"
            lw = 1.8 if is_active else 1.0

            rect = mpatches.FancyBboxPatch(
                (x + pad, y + pad), cell_w - 2*pad, cell_h - 2*pad,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor=ec, linewidth=lw)
            ax.add_patch(rect)

            ax.text(x + 0.5, y + 0.62, sym,
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=AXES_CLR)
            ax.text(x + 0.5, y + 0.28, name,
                    ha="center", va="center", fontsize=7.5,
                    color=AXES_CLR)

    ax.set_title("A Section of the Periodic Table — III-V Optoelectronic Elements",
                 pad=12, fontsize=13)

    fig.tight_layout()
    save(fig, "periodic_table_section.jpg")


fig_02()

# ==============================================================================
#  FIG-6  Current Components Across a p-n Junction
# ==============================================================================
def fig_06():
    x = np.linspace(-5, 5, 600)
    Wp, Wn = -1.0, 1.0

    # Hole current: high in p-region, decays in n-region
    J_hole = np.where(x < Wp, 0.90,
             np.where(x <= Wn, 0.90 - 0.40*(x - Wp)/(Wn - Wp),
                      0.50*np.exp(-(x - Wn)/1.2)))

    # Electron current: complementary so they sum to J_total
    J_total = 0.95
    J_elec  = J_total - J_hole

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # SCL shading
    ax.axvspan(Wp, Wn, color=LAVENDER, alpha=0.10)
    ax.text(0, -0.12, "SCL", ha="center", color=LAVENDER, fontsize=10)

    # Total current
    ax.axhline(J_total, color=AXES_CLR, lw=2.8, ls="-",
               label=r"$J = J_\mathrm{hole} + J_\mathrm{elec}$ (constant)")

    # Hole current
    ax.plot(x, J_hole, color=CORAL, lw=2.5, ls="-", label=r"$J_\mathrm{hole}$")

    # Electron current
    ax.plot(x, J_elec, color=SKYBLUE, lw=2.5, ls="--", label=r"$J_\mathrm{elec}$")

    # Depletion boundaries
    for xd in [Wp, Wn]:
        ax.axvline(xd, color=AXES_CLR, lw=1.0, ls=":")
    ax.text(Wp - 0.1, -0.10, r"$-W_p$", ha="right", color=AXES_CLR, fontsize=10)
    ax.text(Wn + 0.1, -0.10, r"$+W_n$", ha="left",  color=AXES_CLR, fontsize=10)

    # Region labels
    ax.text(-3.5, 1.00, "p-region",  ha="center", color=CORAL,   fontsize=11)
    ax.text( 3.5, 1.00, "n-region",  ha="center", color=SKYBLUE, fontsize=11)

    # Curve annotations
    ax.text(-3.5, 0.70, "Majority\ndrift+diffusion", ha="center", color=CORAL, fontsize=9.5)
    ax.text( 3.5, 0.35, "Minority\ndiffusion", ha="center", color=CORAL, fontsize=9.5)
    ax.text( 3.5, 0.58, "Majority\ndrift+diffusion", ha="center", color=SKYBLUE, fontsize=9.5)

    ax.set_xlabel(r"Position $x$", fontsize=13)
    ax.set_ylabel(r"Current density $J$  (norm.)", fontsize=13)
    ax.set_title(r"Current Components Across a Forward-Biased p-n Junction", pad=10)
    ax.set_xlim(-5, 5); ax.set_ylim(-0.20, 1.15)
    ax.legend(loc="center right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "junction_current_components.jpg")


fig_06()


# ==============================================================================
#  FIG-7  Optical Confinement Factor vs. Active Layer Thickness d
# ==============================================================================
def fig_07():
    z = np.linspace(-2.5, 2.5, 600)   # position um
    d_vals  = [0.1, 0.2, 0.3]         # active layer half-widths (um)
    colors7 = [SKYBLUE, GOLD, CORAL]
    labels7 = [r"$d = 0.1\;\mu$m", r"$d = 0.2\;\mu$m", r"$d = 0.3\;\mu$m"]

    # DH slab waveguide: approximate mode sigma from d and Delta_n=0.1
    # sigma ~ sqrt(d / (2*pi*Delta_n / lambda * d)) — use empirical widths
    sigmas  = [0.80, 0.55, 0.38]   # mode sigma in um

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2.5]})

    # ── Refractive index step profile (top, d=0.3 um shown) ─────────────────
    ax = axes[0]
    d_show = 0.3
    nr = np.where(np.abs(z) <= d_show/2, 3.60, 3.50)
    ax.plot(z, nr, color=AXES_CLR, lw=2.5)
    ax.axvspan(-d_show/2, d_show/2, color=GOLD, alpha=0.15)
    ax.axvline(-d_show/2, color=AXES_CLR, lw=0.9, ls="--")
    ax.axvline( d_show/2, color=AXES_CLR, lw=0.9, ls="--")
    ax.set_ylabel(r"$n_r$", fontsize=12)
    ax.set_ylim(3.46, 3.65)
    ax.set_yticks([3.50, 3.60])
    ax.text(0, 3.61, r"$n = 3.60$", ha="center", color=AXES_CLR, fontsize=10)
    ax.text(1.5, 3.505, r"$\Delta n = 0.10$", color=AXES_CLR, fontsize=10)
    ax.set_title("Optical Mode Confinement vs. Active Layer Width\n"
                 r"($\lambda_L = 0.9\;\mu$m, $\Delta n = 0.1$)", pad=8)
    ax.grid(True)

    # ── Mode intensity profiles (bottom) ────────────────────────────────────
    ax = axes[1]
    for d, sigma, clr, lbl in zip(d_vals, sigmas, colors7, labels7):
        mode = np.exp(-z**2 / (2*sigma**2))
        mode /= mode.max()  # normalize
        # confinement factor
        inside = np.abs(z) <= d/2
        Gamma = np.trapezoid(mode[inside], z[inside]) / np.trapezoid(mode, z)
        ax.plot(z, mode, color=clr, lw=2.5,
                label=rf"{lbl}  ($\Gamma = {Gamma*100:.0f}\%$)")
        ax.axvspan(-d/2, d/2, color=clr, alpha=0.06)

    # Active layer width annotation for d=0.3
    ax.axvline(-0.15, color=CORAL, lw=0.9, ls="--")
    ax.axvline( 0.15, color=CORAL, lw=0.9, ls="--")
    ax.annotate("", xy=(0.15, 0.92), xytext=(-0.15, 0.92),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.3))
    ax.text(0, 0.97, r"$d_3 = 0.3\;\mu$m", ha="center", color=CORAL, fontsize=10)

    ax.set_xlabel(r"Position $x$  ($\mu$m)", fontsize=13)
    ax.set_ylabel(r"Field intensity  $|\mathcal{E}|^2$  (norm.)", fontsize=12)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="upper right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout(h_pad=0.5)
    save(fig, "confinement_vs_thickness.jpg")


fig_07()


# ==============================================================================
#  FIG-8  Gain Spectrum (Left) + Linear Gain Model (Right)
# ==============================================================================
def fig_08():
    Eg = 0.92; kBT = 0.026   # InGaAsP parameters, ~1.3 um

    densities  = [1.0, 1.2, 1.4, 1.6, 1.8]   # x10^18 cm^-3
    Efc_offs   = [0.01, 0.04, 0.08, 0.12, 0.16]
    Efv_offs   = [0.01, 0.04, 0.08, 0.12, 0.16]
    import matplotlib.cm as cm
    colors8 = [cm.coolwarm(i/4) for i in range(5)]

    hnu = np.linspace(0.87, 0.97, 500)

    # Scale G0 from highest density
    def gain_curve(hnu, Eg, kBT, Efc, Efv):
        fc  = 1/(1+np.exp((hnu-Eg-Efc)/kBT))
        fv  = 1/(1+np.exp((hnu-Eg+Efv)/kBT))
        rho = np.sqrt(np.maximum(hnu-Eg, 0))
        return rho*(fc-fv)

    raw_max = max(abs(gain_curve(hnu, Eg, kBT, Efc_offs[-1], Efv_offs[-1])).max(), 1e-12)
    G0 = 250.0 / raw_max

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: gain spectrum ──────────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(0, color=AXES_CLR, lw=1.5)
    ax.axvline(Eg, color=AXES_CLR, lw=1.2, ls="--")
    ax.text(Eg+0.001, -225, r"$E_g$", color=AXES_CLR, fontsize=9.5)
    ax.axhspan(0, 270, color=TEAL,  alpha=0.06)
    ax.axhspan(-270, 0, color=CORAL, alpha=0.06)

    peak_gains = []
    for i, (dn, Efc, Efv, clr) in enumerate(zip(densities, Efc_offs, Efv_offs, colors8)):
        g = G0 * gain_curve(hnu, Eg, kBT, Efc, Efv)
        ax.plot(hnu, g, color=clr, lw=2.2,
                label=rf"$\Delta n={dn}\!\times\!10^{{18}}$")
        peak_gains.append(float(np.max(g)))

    # 75 nm bandwidth annotation
    g_hi = G0 * gain_curve(hnu, Eg, kBT, Efc_offs[-1], Efv_offs[-1])
    zc = np.where(np.diff(np.sign(g_hi)))[0]
    if len(zc) >= 2:
        x1, x2 = hnu[zc[0]], hnu[zc[-1]]
        ax.annotate("", xy=(x2, 28), xytext=(x1, 28),
                    arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.2))
        ax.text((x1+x2)/2, 45,
                f"~{abs(x2-x1)*1240:.0f} nm BW",
                ha="center", color=AXES_CLR, fontsize=9.5)

    ax.set_xlabel(r"Photon energy $h\nu$  (eV)", fontsize=12)
    ax.set_ylabel(r"Gain $\gamma$  (cm$^{-1}$)", fontsize=12)
    ax.set_title(r"(a) Gain Spectrum — InGaAsP ($\lambda = 1.3\;\mu$m)", pad=8)
    ax.set_xlim(0.87, 0.97); ax.set_ylim(-260, 260)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    # ── Right: peak gain vs carrier density (linear model) ──────────────────
    ax = axes[1]
    N_arr = np.array(densities)
    peak_arr = np.array(peak_gains)
    peak_arr_plot = np.maximum(peak_arr, 0)   # only positive gains shown

    N_tr = 1.0   # transparency density

    # Full curve
    ax.plot(N_arr, peak_arr_plot, color=AXES_CLR, lw=2.5, marker="o", ms=7)

    # Linear tangent fit
    B_fit = (peak_arr_plot[-1] - 0) / (N_arr[-1] - N_tr)
    N_fit = np.linspace(N_tr, 2.1, 100)
    ax.plot(N_fit, B_fit*(N_fit - N_tr), color=CORAL, lw=2.0, ls="--",
            label=r"$B(N - N_{tr})$")

    ax.axvline(N_tr, color=AXES_CLR, lw=1.0, ls=":")
    ax.text(N_tr+0.03, 10, r"$N_{tr}$", color=AXES_CLR, fontsize=10)
    ax.text(1.50, 80, r"InGaAsP" "\n" r"$\lambda = 1.3\;\mu$m",
            color=AXES_CLR, fontsize=10)

    ax.set_xlabel(r"Carrier density $N$  ($10^{18}\;\mathrm{cm}^{-3}$)", fontsize=12)
    ax.set_ylabel(r"Peak gain (cm$^{-1}$)", fontsize=12)
    ax.set_title(r"(b) Peak Gain vs. Carrier Density", pad=8)
    ax.set_xlim(0.8, 2.1); ax.set_ylim(0, 310)
    ax.legend(loc="upper left", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout(pad=2.0)
    save(fig, "gain_spectrum_and_linear_model.jpg")


fig_08()
