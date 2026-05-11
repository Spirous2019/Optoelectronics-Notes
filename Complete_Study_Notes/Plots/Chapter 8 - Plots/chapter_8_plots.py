"""
Chapter 8 Plots — LED, SOA, and Laser Diode
Python figures: 2, 3, 4, 5, 8, 10
AI-generated figures: 1 (escape cone), 6 (SOA vs FP), 7 (facet geometries), 9 (FP block diagram)
— see Chapter 8 Plots.md for AI prompts.
Output: Chapter 8 - Plots/ (same directory as this script)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

WHITE    = "#ffffff"; AXES_CLR = "#2b2b2b"; GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"; CORAL    = "#d94452"; GOLD     = "#c28800"
LAVENDER = "#7e57c2"; SKYBLUE  = "#1976d2"
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
#  FIG-2  LED P-I Characteristic at Multiple Temperatures
# ==============================================================================
def fig_02():
    I = np.linspace(0, 150, 500)   # mA

    def led_pi(I, Pmax, I_rolloff):
        """Smooth sub-linear rolloff: P = Pmax * tanh(I / I_rolloff)."""
        return Pmax * np.tanh(I / I_rolloff)

    temps   = [0,   25,   70]
    Pmaxes  = [3.5, 3.0,  2.2]
    Irolls  = [60,  55,   42]
    colors  = [SKYBLUE, TEAL, CORAL]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for T, Pmax, Iroll, clr in zip(temps, Pmaxes, Irolls, colors):
        P = led_pi(I, Pmax, Iroll)
        ax.plot(I, P, color=clr, lw=2.5,
                label=rf"$T = {T}\,^\circ$C")

    ax.set_xlabel("Drive current $I$  (mA)", fontsize=13)
    ax.set_ylabel("Output power $P$  (mW)", fontsize=13)
    ax.set_title("LED P\u2013I Characteristics at Multiple Temperatures", pad=10)
    ax.set_xlim(0, 150); ax.set_ylim(0, 4.0)
    ax.legend(loc="lower right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "led_pi_temperature.jpg")


fig_02()


# ==============================================================================
#  FIG-3  LED Emission Spectrum (Two-Panel)
# ==============================================================================
def fig_03():
    kBT = 0.026   # eV at 300K

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # ── Top: Normalised spontaneous emission lineshape ───────────────────────
    ax = axes[0]
    Eg = 0.0   # reference energy; plot relative to Eg
    hnu = np.linspace(-0.02, 0.20, 600)

    # R_sp ~ (hnu-Eg)^0.5 * exp(-(hnu-Eg)/kBT) for hnu > Eg
    Rsp = np.where(hnu >= 0,
                   np.sqrt(np.maximum(hnu, 0)) * np.exp(-hnu / kBT),
                   0.0)
    Rsp /= Rsp.max()

    ax.plot(hnu, Rsp, color=AXES_CLR, lw=2.5)
    ax.axhline(0, color=AXES_CLR, lw=0.8)
    ax.axvline(0, color=AXES_CLR, lw=1.0, ls="--")

    # FWHM annotation (~1.8 kBT)
    fwhm_hw = 0.9 * kBT
    ax.axhline(0.5, color=AXES_CLR, lw=1.0, ls="--")
    ax.annotate("", xy=(fwhm_hw, 0.5), xytext=(-fwhm_hw, 0.5),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
    ax.text(0, 0.56, r"$\sim 1.8\,k_BT$", ha="center", color=AXES_CLR, fontsize=10)

    # Peak annotation
    peak_hnu = 0.5 * kBT
    ax.axvline(peak_hnu, color=AXES_CLR, lw=0.7, ls=":")
    ax.text(peak_hnu + 0.002, 1.04,
            r"$E_g + \frac{1}{2}k_BT$", color=AXES_CLR, fontsize=10)

    # x-axis tick labels
    ax.set_xticks([0, 0.5*kBT, kBT, 2*kBT])
    ax.set_xticklabels([r"$E_g$", r"$h\nu_2$", r"$h\nu_3$", ""], fontsize=10)
    ax.text(-0.012, -0.07, r"$h\nu_1$", ha="right", fontsize=10, color=AXES_CLR)

    ax.set_ylabel(r"Normalised $R_{sp}$", fontsize=12)
    ax.set_title(r"(a) Spontaneous Emission Lineshape", pad=8)
    ax.set_xlim(-0.02, 0.20); ax.set_ylim(-0.05, 1.15)
    ax.grid(True)

    # ── Bottom: Measured GaAs LED spectrum ──────────────────────────────────
    ax = axes[1]
    lam = np.linspace(760, 880, 600)   # nm
    lam0 = 822.0; dlam = 20.0         # peak and half-width
    spectrum = np.exp(-(lam - lam0)**2 / (2*dlam**2))

    ax.plot(lam, spectrum, color=AXES_CLR, lw=2.5)
    ax.axhline(0.5, color=AXES_CLR, lw=1.0, ls="--")
    ax.axvline(lam0, color=AXES_CLR, lw=0.9, ls=":")
    ax.text(lam0 + 2, 1.02, "822 nm", color=AXES_CLR, fontsize=10)

    # Delta-lambda annotation
    lam_lo = lam0 - 20; lam_hi = lam0 + 20
    ax.annotate("", xy=(lam_hi, 0.5), xytext=(lam_lo, 0.5),
                arrowprops=dict(arrowstyle="<->", color=AXES_CLR, lw=1.3))
    ax.text(lam0, 0.57, r"$\Delta\lambda \approx 40\,\mathrm{nm}$",
            ha="center", color=AXES_CLR, fontsize=10)

    ax.set_xlabel("Wavelength  (nm)", fontsize=12)
    ax.set_ylabel("Relative spectral power", fontsize=12)
    ax.set_title("(b) Measured GaAs LED Spectrum", pad=8)
    ax.set_xlim(760, 880); ax.set_ylim(-0.05, 1.15)
    ax.grid(True)

    fig.tight_layout(h_pad=1.5)
    save(fig, "led_emission_spectrum.jpg")


fig_03()


# ==============================================================================
#  FIG-4  Photopic and Scotopic Luminosity Functions V(lambda)
# ==============================================================================
def fig_04():
    lam = np.linspace(380, 780, 800)

    # CIE 1931 photopic V(lambda) — Gaussian approximation
    V_phot  = np.exp(-0.5*((lam - 555)/55)**2)

    # CIE scotopic V'(lambda) — shifted to 507 nm
    V_scot  = np.exp(-0.5*((lam - 507)/55)**2)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Visible spectrum rainbow background
    color_bands = [
        (380, 420, "#8B00FF"),  # violet
        (420, 460, "#4400CC"),  # indigo
        (460, 490, "#0000FF"),  # blue
        (490, 560, "#00AA00"),  # green
        (560, 590, "#AAAA00"),  # yellow
        (590, 625, "#FF6600"),  # orange
        (625, 700, "#CC0000"),  # red
    ]
    for (lam_lo, lam_hi, clr) in color_bands:
        ax.axvspan(lam_lo, lam_hi, color=clr, alpha=0.13)
        ax.text((lam_lo+lam_hi)/2, 1.04,
                {380: "V", 420: "I", 460: "B", 490: "G",
                 560: "Y", 590: "O", 625: "R"}.get(lam_lo, ""),
                ha="center", color=clr, fontsize=9, fontweight="bold")

    ax.plot(lam, V_phot, color=AXES_CLR, lw=2.8, label=r"Photopic $V(\lambda)$")
    ax.plot(lam, V_scot, color="#888888", lw=2.2, ls="--", label=r"Scotopic $V'(\lambda)$")

    ax.axvline(555, color=AXES_CLR, lw=0.9, ls=":", alpha=0.6)
    ax.axvline(507, color="#888888", lw=0.9, ls=":", alpha=0.6)
    ax.text(557, 0.85, "555 nm", color=AXES_CLR, fontsize=9.5)
    ax.text(509, 0.85, "507 nm", color="#888888", fontsize=9.5)

    ax.set_xlabel(r"Wavelength  (nm)", fontsize=13)
    ax.set_ylabel(r"Relative luminous efficiency", fontsize=13)
    ax.set_title(r"Photopic and Scotopic Luminosity Functions $V(\lambda)$", pad=10)
    ax.set_xlim(380, 780); ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "luminosity_functions.jpg")


fig_04()


# ==============================================================================
#  FIG-5  SOA Gain Saturation: Gain vs. Input Power
# ==============================================================================
def fig_05():
    P_in_dBm = np.linspace(-60, 10, 500)
    P_in_lin  = 10**(P_in_dBm / 10)   # mW (linear)

    G0   = 30.0    # dB small-signal gain
    G0_lin = 10**(G0/10)
    P_sat_dBm = -30   # saturation input power (dBm)
    P_sat_lin = 10**(P_sat_dBm/10)

    # g = G0 / (1 + P/P_s) model — convert to dB
    G_dB = 10*np.log10(G0_lin / (1 + P_in_lin / P_sat_lin))
    G_dB = np.clip(G_dB, 0, G0 + 2)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(P_in_dBm, G_dB, color=SKYBLUE, lw=2.8)

    # Reference lines
    ax.axhline(G0,    color=AXES_CLR, lw=1.2, ls="--")
    ax.axhline(G0-3,  color=AXES_CLR, lw=1.0, ls="--")
    ax.axvline(P_sat_dBm, color=CORAL, lw=1.2, ls=":")

    ax.text(-58, G0+0.8, r"$G_0 = 30\,\mathrm{dB}$", color=AXES_CLR, fontsize=10)
    ax.text(-58, G0-3+0.8, r"$G_0 - 3\,\mathrm{dB}$", color=AXES_CLR, fontsize=10)
    ax.text(P_sat_dBm+0.5, 2, r"$P_{amp,sat}$", color=CORAL, fontsize=10, rotation=90, va="bottom")
    ax.text(-10, 12, "Saturation\nregion", color=CORAL, fontsize=10, ha="center")

    ax.set_xlabel(r"Input signal power  (dBm)", fontsize=13)
    ax.set_ylabel(r"Gain  (dB)", fontsize=13)
    ax.set_title(r"SOA Gain Saturation: $G = G_0\,/\,(1 + P_{in}/P_s)$", pad=10)
    ax.set_xlim(-60, 10); ax.set_ylim(0, 35)
    ax.grid(True)

    fig.tight_layout()
    save(fig, "soa_gain_saturation.jpg")


fig_05()


# ==============================================================================
#  FIG-8  SOA Gain Spectrum with Fabry-Perot Ripple (TE)
# ==============================================================================
def fig_08():
    lam = np.linspace(1.44, 1.54, 2000)   # um

    # Smooth Gaussian envelope peaking at 1.50 um
    lam0 = 1.50; dlam = 0.025
    envelope = 25 * np.exp(-((lam - lam0)**2) / (2*dlam**2))

    # FP ripple: sinusoidal with period ~ 0.5 nm (dense)
    ripple_period = 0.0005   # um
    ripple_amp = 1.8
    ripple = ripple_amp * np.sin(2*np.pi*lam / ripple_period)

    G_TE = envelope + ripple
    G_TE = np.clip(G_TE, 0, 31)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(lam, G_TE, color=AXES_CLR, lw=1.0)
    ax.plot(lam, envelope, color=TEAL, lw=1.5, ls="--", alpha=0.6, label="Smooth envelope")
    ax.text(1.497, 26.5, "TE", color=AXES_CLR, fontsize=11, fontweight="bold")

    ax.set_xlabel(r"Wavelength  ($\mu$m)", fontsize=13)
    ax.set_ylabel(r"Signal gain $G$  (dB)", fontsize=13)
    ax.set_title("SOA Gain Spectrum — TE Polarization (with FP ripple)", pad=10)
    ax.set_xlim(1.44, 1.54); ax.set_ylim(0, 32)
    ax.legend(loc="lower right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "soa_gain_spectrum_te.jpg")


fig_08()


# ==============================================================================
#  FIG-10  Laser Diode DC Characteristics: N and S vs. Current
# ==============================================================================
def fig_10():
    J = np.linspace(0, 3.0, 500)   # normalised current (Jth = 1)
    Jth = 1.0

    # Carrier density: linear ramp then clamped
    N = np.where(J <= Jth, J, Jth)
    Nth = Jth

    # Photon density: ~zero below threshold, linear above
    # Include small spontaneous emission floor below threshold
    beta_factor = 0.02
    S_sp  = beta_factor * N    # spontaneous coupling (small floor)
    S_stim = np.where(J <= Jth, 0.0, (J - Jth) * 1.5)
    S = S_sp + S_stim

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ── Top: N vs J ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(J, N, color=SKYBLUE, lw=2.5)
    ax.axvline(Jth, color=CORAL, lw=1.5, ls="--")
    ax.axhline(Nth, color=AXES_CLR, lw=1.0, ls="--")
    ax.text(Jth + 0.04, -0.08, r"$J_{th}$", color=CORAL, fontsize=11)
    ax.text(-0.08, Nth + 0.02, r"$N_{th}$", color=AXES_CLR, fontsize=11, ha="right")
    ax.set_ylabel(r"Carrier density $N$  (norm.)", fontsize=12)
    ax.set_title("(a) Carrier Density vs. Injection Current", pad=8)
    ax.set_ylim(-0.05, 1.30)
    ax.set_yticks([0, Nth])
    ax.set_yticklabels(["0", r"$N_{th}$"])
    ax.grid(True)
    ax.text(2.0, 0.85, "Carrier\nclamping", ha="center", color=SKYBLUE, fontsize=10)

    # ── Bottom: S vs J ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(J, S, color=SKYBLUE, lw=2.5)
    ax.plot(J, S_sp, color=AXES_CLR, lw=1.2, ls=":", alpha=0.7,
            label=r"Spontaneous floor $\beta\tau_{ph}N/\tau_r$")
    ax.axvline(Jth, color=CORAL, lw=1.5, ls="--")
    ax.text(Jth + 0.04, -0.05, r"$J_{th}$", color=CORAL, fontsize=11)

    # Slope annotation above threshold
    J_ann1, J_ann2 = 1.8, 2.8
    S_ann1 = float(np.interp(J_ann1, J, S)); S_ann2 = float(np.interp(J_ann2, J, S))
    ax.annotate("", xy=(J_ann2, S_ann2), xytext=(J_ann1, S_ann1),
                arrowprops=dict(arrowstyle="->", color=AXES_CLR, lw=1.3))
    ax.text((J_ann1+J_ann2)/2 + 0.12, (S_ann1+S_ann2)/2,
            r"slope $= \tau_{ph}/ed$",
            color=AXES_CLR, fontsize=10)

    ax.set_xlabel(r"Injection current $J$  (norm.)", fontsize=12)
    ax.set_ylabel(r"Photon density $S$  (norm.)", fontsize=12)
    ax.set_title("(b) Photon Density vs. Injection Current", pad=8)
    ax.set_xlim(0, 3.0); ax.set_ylim(-0.10, S.max()*1.15)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout(h_pad=0.8)
    save(fig, "ld_dc_characteristics.jpg")


fig_10()
