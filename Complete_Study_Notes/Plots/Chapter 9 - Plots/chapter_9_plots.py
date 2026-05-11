"""
Chapter 9 Plots — LD Static Response and Modulation
Python figures: 2, 3, 4, 5, 6, 7, 8, 9, 10
AI-generated figures: 1 (FP cavity round-trip schematic)
— see Chapter 9 Plots.md for the AI prompt.
Output: Chapter 9 - Plots/ (same directory as this script)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import os

WHITE    = "#ffffff"; AXES_CLR = "#2b2b2b"; GRID_CLR = "#d0d0d0"
TEAL     = "#0d9b76"; CORAL    = "#d94452"; GOLD     = "#c28800"
LAVENDER = "#7e57c2"; SKYBLUE  = "#1976d2"; ORANGE   = "#e65100"

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
#  FIG-2  Carrier Density Below Threshold
# ==============================================================================
def fig_02():
    J    = np.linspace(0, 1.5, 500)   # normalised, Jth = 1
    Jth  = 1.0
    Nth  = 1.0   # normalised Nth

    N = np.minimum(J * Nth / Jth, Nth)   # linear ramp clamped at Nth

    fig, ax = plt.subplots(figsize=(8, 5))

    # Below-threshold shading
    ax.axvspan(0, Jth, color=SKYBLUE, alpha=0.07, label="Below threshold")

    ax.plot(J, N, color=SKYBLUE, lw=2.5)

    ax.axhline(Nth, color=CORAL, lw=1.4, ls="--")
    ax.axvline(Jth, color=CORAL, lw=1.4, ls="--")

    # Equation label on the line
    ax.text(0.42, 0.55, r"$N = \dfrac{J\,\tau}{e d}$",
            color=SKYBLUE, fontsize=12, rotation=40,
            ha="center", va="bottom")

    ax.text(-0.04, Nth + 0.03, r"$N_{th}$", ha="right", color=CORAL, fontsize=11)
    ax.text(Jth, -0.08, r"$J_{th}$", ha="center", color=CORAL, fontsize=11)
    ax.text(0.45, -0.13, "below threshold", ha="center", color=SKYBLUE,
            fontsize=9.5, alpha=0.85)

    ax.set_xlabel(r"Injection current density $J$  (norm.)", fontsize=13)
    ax.set_ylabel(r"Carrier density $N$  (norm.)", fontsize=13)
    ax.set_title("Carrier Density Build-Up Below Threshold", pad=10)
    ax.set_xlim(0, 1.5); ax.set_ylim(-0.15, 1.35)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)

    fig.tight_layout()
    save(fig, "carrier_density_below_threshold.jpg")


fig_02()


# ==============================================================================
#  FIG-3  Carrier Clamping and Photon Build-Up Above Threshold
# ==============================================================================
def fig_03():
    J   = np.linspace(0, 3.0, 500)
    Jth = 1.0; Nth = 1.0; beta = 0.02

    N    = np.where(J <= Jth, J, Jth)
    S_sp = beta * N
    S    = S_sp + np.where(J <= Jth, 0.0, (J - Jth) * 1.5)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ── Top ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(J, N, color=SKYBLUE, lw=2.5)
    ax.axvline(Jth, color=CORAL, lw=1.5, ls="--")
    ax.axhline(Nth, color=AXES_CLR, lw=1.0, ls="--")
    ax.text(Jth + 0.05, -0.07, r"$J_{th}$", color=CORAL, fontsize=11)
    ax.text(-0.08, Nth + 0.03, r"$N_{th}$", ha="right", color=AXES_CLR, fontsize=11)
    ax.text(2.0, 0.82, "carrier\nclamping", ha="center", color=SKYBLUE, fontsize=10)
    ax.set_ylabel(r"Carrier density $N$", fontsize=12)
    ax.set_title("(a) Carrier Clamping", pad=8)
    ax.set_ylim(-0.05, 1.30)
    ax.set_yticks([]); ax.grid(False)

    # ── Bottom ────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(J, S, color=SKYBLUE, lw=2.5)
    ax.plot(J, S_sp, color=AXES_CLR, lw=1.2, ls=":", alpha=0.7,
            label=r"Spontaneous floor $\beta\tau_{ph}N_{th}/\tau_r$")
    ax.axvline(Jth, color=CORAL, lw=1.5, ls="--")
    ax.text(Jth + 0.05, -0.06, r"$J_{th}$", color=CORAL, fontsize=11)

    # Slope label
    J1, J2 = 1.8, 2.8
    S1, S2 = float(np.interp(J1, J, S)), float(np.interp(J2, J, S))
    ax.annotate("", xy=(J2, S2), xytext=(J1, S1),
                arrowprops=dict(arrowstyle="->", color=AXES_CLR, lw=1.3))
    ax.text((J1+J2)/2 + 0.14, (S1+S2)/2,
            r"slope $= \tau_{ph}/ed$", color=AXES_CLR, fontsize=10)

    ax.set_xlabel(r"Injection current density $J$  (norm.)", fontsize=12)
    ax.set_ylabel(r"Photon density $S$", fontsize=12)
    ax.set_title("(b) Photon Build-Up Above Threshold", pad=8)
    ax.set_xlim(0, 3.0); ax.set_ylim(-0.10, S.max()*1.15)
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9,
              facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(False)

    fig.tight_layout(h_pad=0.8)
    save(fig, "carrier_clamping_photon_growth.jpg")


fig_03()


# ==============================================================================
#  FIG-4  Laser Output Power vs. Drive Current (P-I Characteristic)
# ==============================================================================
def fig_04():
    I   = np.linspace(0, 3.0, 500)
    Ith = 1.0; eta_d = 1.0; Ps = 0.02; beta = 0.008

    # Spontaneous emission (very weak below threshold)
    P_sp = beta * I * 0.04
    # Stimulated above threshold
    P_stim = np.where(I <= Ith, 0.0, eta_d*(I - Ith) + Ps)
    P = P_sp + P_stim

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(I, P, color=AXES_CLR, lw=2.5)
    ax.axvline(Ith, color=CORAL, lw=1.4, ls="--")
    ax.text(Ith + 0.04, -0.08, r"$I_{th}$", color=CORAL, fontsize=11)

    # Slope annotation
    I_a, I_b = 1.8, 2.8
    P_a, P_b = float(np.interp(I_a, I, P)), float(np.interp(I_b, I, P))
    ax.annotate("", xy=(I_b, P_b), xytext=(I_a, P_a),
                arrowprops=dict(arrowstyle="->", color=AXES_CLR, lw=1.3))
    ax.text((I_a+I_b)/2 + 0.12, (P_a+P_b)/2,
            r"slope $= \eta_d$", color=AXES_CLR, fontsize=11)

    # Label line
    ax.text(2.2, eta_d*(2.2 - Ith) + Ps + 0.12,
            r"$P = \eta_d(I - I_{th}) + P_s$",
            color=AXES_CLR, fontsize=10, ha="center")

    ax.set_xlabel(r"Drive current $I$  (norm.)", fontsize=13)
    ax.set_ylabel(r"Output power $P$  (norm.)", fontsize=13)
    ax.set_title(r"Laser Diode P–I Characteristic and Slope Efficiency", pad=10)
    ax.set_xlim(0, 3.0); ax.set_ylim(-0.15, P.max()*1.20)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)

    fig.tight_layout()
    save(fig, "ld_pi_characteristic.jpg")


fig_04()


# ==============================================================================
#  FIG-5  Temperature Dependence of Laser P-I Curves
# ==============================================================================
def fig_05():
    I    = np.linspace(0, 3.5, 500)
    T0   = 50.0   # characteristic temperature (K)
    T_ref = 300.0

    temps     = [300, 320, 340, 360]   # K
    Iths_ref  = 1.0
    labels    = [r"$T_1$", r"$T_2$", r"$T_3$", r"$T_4$"]
    colors5   = [SKYBLUE, TEAL, GOLD, CORAL]

    fig, ax = plt.subplots(figsize=(9, 6))

    for T, lbl, clr in zip(temps, labels, colors5):
        Ith_T = Iths_ref * np.exp((T - T_ref) / T0)
        eta_T = 0.95 / (1 + 0.008*(T - T_ref))   # slight slope reduction
        P_sp  = 0.008 * I
        P     = P_sp + np.where(I <= Ith_T, 0.0, eta_T*(I - Ith_T))
        ax.plot(I, P, color=clr, lw=2.5, label=f"{lbl}  ($T = {T}$ K)")

        # Threshold dot
        ax.plot(Ith_T, 0.0, "o", ms=6, color=clr, zorder=5)

    # Threshold drift arrow
    Ith_low  = Iths_ref * np.exp((temps[0]  - T_ref)/T0)
    Ith_high = Iths_ref * np.exp((temps[-1] - T_ref)/T0)
    ax.annotate("", xy=(Ith_high, -0.10), xytext=(Ith_low, -0.10),
                arrowprops=dict(arrowstyle="->", color=AXES_CLR, lw=1.4))
    ax.text((Ith_low+Ith_high)/2, -0.18, r"$I_{th}$ shifts right with $T$",
            ha="center", color=AXES_CLR, fontsize=9.5)

    # Equation inset
    ax.text(0.6, P.max()*0.85,
            r"$I_{th}(T) \propto e^{(T-T_{ref})/T_0}$",
            color=AXES_CLR, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#cccccc", lw=0.8))

    ax.set_xlabel(r"Drive current $I$  (norm.)", fontsize=13)
    ax.set_ylabel(r"Output power $P$  (norm.)", fontsize=13)
    ax.set_title("Temperature Dependence of Laser P–I Curves", pad=10)
    ax.set_xlim(0, 3.5); ax.set_ylim(-0.25, P.max()*1.20)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(False)

    fig.tight_layout()
    save(fig, "ld_pi_temperature.jpg")


fig_05()


# ==============================================================================
#  FIG-6  Small-Signal Modulation Transfer Function |H(jw)|
# ==============================================================================
def fig_06():
    f    = np.logspace(7, 12, 1000)   # Hz
    wr   = 2*np.pi * 5e9              # relaxation resonance ~5 GHz
    H0   = 1.0
    kBT  = 0.026

    # Two damping levels
    damping_configs = [
        (0.15, SKYBLUE, r"Low damping ($\xi$ small)"),
        (0.55, CORAL,   r"High damping ($\xi$ large)"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for (xi, clr, lbl) in damping_configs:
        omega = 2*np.pi*f
        H = H0 * wr**2 / np.sqrt((wr**2 - omega**2)**2 + (2*xi*wr*omega)**2)
        ax.semilogx(f/1e9, 20*np.log10(H), color=clr, lw=2.5, label=lbl)

    # Annotations
    fr_GHz = wr / (2*np.pi) / 1e9
    ax.axvline(fr_GHz, color=AXES_CLR, lw=1.2, ls="--")
    ax.text(fr_GHz*1.05, -2,
            r"$f_r = \frac{\omega_r}{2\pi}$",
            color=AXES_CLR, fontsize=10)

    ax.axhline(0, color=AXES_CLR, lw=0.9)
    ax.text(0.012, 1.5, r"$H_0 = \tau_{ph}/ed$", color=AXES_CLR, fontsize=10)

    ax.set_xlabel(r"Modulation frequency  (GHz)", fontsize=13)
    ax.set_ylabel(r"$|H(j\omega)|$  (dB, norm.)", fontsize=13)
    ax.set_title("Small-Signal Modulation Transfer Function", pad=10)
    ax.set_xlim(f[0]/1e9, f[-1]/1e9)
    ax.set_ylim(-40, 12)
    ax.legend(loc="lower left", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "modulation_transfer_function.jpg")


fig_06()


# ==============================================================================
#  FIG-7  Turn-On Delay After a Current Step
# ==============================================================================
def fig_07():
    t     = np.linspace(-0.5, 4.0, 1000)   # ns
    J1    = 0.5   # below threshold (normalised)
    J2    = 2.0   # above threshold
    Jth   = 1.0
    tau_n = 1.0   # ns (carrier lifetime)

    # Turn-on delay: td = tau * ln((J2-J1)/(J2-Jth))
    td = tau_n * np.log((J2 - J1) / (J2 - Jth))

    # Current step
    J_t = np.where(t < 0, J1, J2)

    # Photon output: ~0 for t < td, then rises (simplified exponential rise)
    def S_out(t):
        S = np.zeros_like(t)
        mask = t >= td
        S[mask] = (J2 - Jth) * (1 - np.exp(-(t[mask] - td) / 0.3))
        # Optional overshoot
        S[mask] += 0.4*(J2 - Jth)*np.exp(-(t[mask]-td)/0.5)*np.sin(2*np.pi*(t[mask]-td)/0.6)
        return np.maximum(S, 0)

    S_t = S_out(t)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # ── Top: Current step ────────────────────────────────────────────────────
    ax = axes[0]
    ax.step(t, J_t, color=AXES_CLR, lw=2.5, where="post")
    ax.axhline(Jth, color=CORAL, lw=1.3, ls="--")
    ax.text(3.8, Jth + 0.05, r"$J_{th}$", ha="right", color=CORAL, fontsize=11)
    ax.text(3.8, J1 + 0.05, r"$J_1$", ha="right", color=AXES_CLR, fontsize=10)
    ax.text(3.8, J2 + 0.05, r"$J_2$", ha="right", color=AXES_CLR, fontsize=10)
    ax.set_ylabel(r"Current density $J$  (norm.)", fontsize=12)
    ax.set_title("(a) Current Step", pad=8)
    ax.set_ylim(0, 2.6); ax.grid(True)
    ax.set_yticks([])

    # ── Bottom: Optical response ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, S_t, color=SKYBLUE, lw=2.5)
    ax.axvline(0,  color=AXES_CLR, lw=1.0, ls=":")
    ax.axvline(td, color=CORAL, lw=1.5, ls="--")

    # Delay bracket
    ax.annotate("", xy=(td, -0.12), xytext=(0, -0.12),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.4))
    ax.text(td/2, -0.22, r"$t_d$", ha="center", color=CORAL, fontsize=11)

    # Formula box
    ax.text(2.5, S_t.max()*0.55,
            r"$t_d = \tau \ln\!\left[\dfrac{J_2 - J_1}{J_2 - J_{th}}\right]$",
            color=AXES_CLR, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#cccccc", lw=0.8))

    ax.set_xlabel(r"Time  (ns, norm.)", fontsize=12)
    ax.set_ylabel(r"Photon density $S$  (norm.)", fontsize=12)
    ax.set_title("(b) Optical Response: Turn-On Delay", pad=8)
    ax.set_xlim(-0.5, 4.0); ax.set_ylim(-0.30, S_t.max()*1.25)
    ax.grid(True); ax.set_yticks([])

    fig.tight_layout(h_pad=0.8)
    save(fig, "turnon_delay.jpg")


fig_07()


# ==============================================================================
#  FIG-8  Relaxation Oscillations in Time Domain
# ==============================================================================
def fig_08():
    t  = np.linspace(0, 5, 1000)   # ns (normalised)
    wr = 2*np.pi * 1.0              # normalised relaxation frequency
    alpha_d = 0.5                   # damping coefficient

    # Damped sinusoidal responses
    Omega  = np.sqrt(max(wr**2 - alpha_d**2, 0))
    dN     = np.exp(-alpha_d*t) * np.sin(Omega*t)
    dS     = np.exp(-alpha_d*t) * np.cos(Omega*t)
    env    = np.exp(-alpha_d*t)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, y, lbl, clr, title in zip(
        axes,
        [dN, dS],
        [r"$\Delta N(t)$", r"$\Delta S(t)$"],
        [CORAL, SKYBLUE],
        [r"(a) Carrier Density Perturbation $\Delta N(t)$",
         r"(b) Photon Density Perturbation $\Delta S(t)$"]
    ):
        ax.fill_between(t, 0, y, color=clr, alpha=0.12)
        ax.plot(t, y, color=clr, lw=2.5, label=lbl)
        ax.plot(t,  env, color="#888888", lw=1.4, ls="--", label=r"Envelope $e^{-\alpha t}$")
        ax.plot(t, -env, color="#888888", lw=1.4, ls="--")
        ax.axhline(0, color=AXES_CLR, lw=0.8)
        ax.set_ylabel(lbl, fontsize=12)
        ax.set_title(title, pad=8)
        ax.set_ylim(-1.25, 1.25); ax.grid(True); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9,
                  facecolor="#f5f5f5", edgecolor="#cccccc")

    # Frequency label
    axes[0].text(0.5/wr*np.pi, 0.90,
                 r"$\Omega = \sqrt{\omega_0^2 - \alpha^2}$",
                 color=AXES_CLR, fontsize=10)

    axes[1].set_xlabel(r"Time  (norm.)", fontsize=12)
    fig.suptitle("Relaxation Oscillations in the Time Domain", y=1.01, fontsize=14)
    fig.tight_layout(h_pad=0.8)
    save(fig, "relaxation_oscillations.jpg")


fig_08()


# ==============================================================================
#  FIG-9  Generic Second-Order System: Underdamped, Critical, Overdamped
# ==============================================================================
def fig_09():
    t  = np.linspace(0, 6, 800)   # normalised time (t * wn)
    wn = 1.0

    damping_cases = [
        (0.25,  SKYBLUE, r"Underdamped ($\delta = 0.25$)"),
        (1.00,  TEAL,    r"Critically damped ($\delta = 1.0$)"),
        (2.50,  CORAL,   r"Overdamped ($\delta = 2.5$)"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(1.0, color=AXES_CLR, lw=0.9, ls="--", alpha=0.5)  # final value

    for (delta, clr, lbl) in damping_cases:
        # Step response of H(s) = wn^2 / (s^2 + 2*delta*wn*s + wn^2)
        system = sig.TransferFunction([wn**2], [1, 2*delta*wn, wn**2])
        t_out, y_out = sig.step(system, T=t)
        ax.plot(t_out, y_out, color=clr, lw=2.5, label=lbl)

    ax.text(0.3, 1.05, "Steady state = 1", color=AXES_CLR, fontsize=9.5, alpha=0.7)

    # Equation box
    ax.text(4.2, 0.22,
            r"$H(s)=\dfrac{1}{1+\frac{2\delta}{\omega_n}s+\frac{s^2}{\omega_n^2}}$",
            color=AXES_CLR, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#cccccc", lw=0.8))

    ax.set_xlabel(r"Normalised time $\omega_n t$", fontsize=13)
    ax.set_ylabel("Step response amplitude", fontsize=13)
    ax.set_title("Second-Order System: Step Response for Different Damping Ratios", pad=10)
    ax.set_xlim(0, 6); ax.set_ylim(-0.15, 1.55)
    ax.legend(loc="lower right", framealpha=0.9, facecolor="#f5f5f5", edgecolor="#cccccc")
    ax.grid(True)

    fig.tight_layout()
    save(fig, "second_order_damping.jpg")


fig_09()


# ==============================================================================
#  FIG-10  Relaxation Frequency vs. Bias Current
# ==============================================================================
def fig_10():
    m = np.linspace(1.01, 5.0, 400)   # m = I/Ith, above threshold

    # wr^2 proportional to (m-1)
    tau_ph = 1.0; tau_n = 1.0
    wr2    = (m - 1) / (tau_ph * tau_n)
    wr     = np.sqrt(wr2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: wr^2 vs m (linear)
    ax = axes[0]
    ax.plot(m, wr2, color=SKYBLUE, lw=2.5)
    ax.set_xlabel(r"Normalised bias $m = I/I_{th}$", fontsize=12)
    ax.set_ylabel(r"$\omega_r^2$  (norm.)", fontsize=12)
    ax.set_title(r"(a) $\omega_r^2 \propto (m-1)$ — Linear regime", pad=8)
    ax.set_xlim(1, 5); ax.set_ylim(0, wr2.max()*1.10)
    ax.grid(True)

    # Slope annotation
    m1, m2 = 2.0, 4.0
    w1, w2 = float(np.interp(m1, m, wr2)), float(np.interp(m2, m, wr2))
    ax.annotate("", xy=(m2, w2), xytext=(m1, w1),
                arrowprops=dict(arrowstyle="->", color=AXES_CLR, lw=1.3))
    ax.text((m1+m2)/2 + 0.15, (w1+w2)/2,
            r"slope $= 1/(\tau_{ph}\tau)$", color=AXES_CLR, fontsize=9.5)

    # Right: fr vs m (square root curve)
    ax = axes[1]
    ax.plot(m, wr/(2*np.pi), color=SKYBLUE, lw=2.5)
    ax.set_xlabel(r"Normalised bias $m = I/I_{th}$", fontsize=12)
    ax.set_ylabel(r"Relaxation frequency $f_r$  (norm.)", fontsize=12)
    ax.set_title(r"(b) $f_r \propto \sqrt{m-1}$", pad=8)
    ax.set_xlim(1, 5); ax.set_ylim(0, wr.max()/(2*np.pi)*1.15)
    ax.grid(True)

    # Annotation
    ax.text(3.5, wr[-1]/(2*np.pi)*0.40,
            r"$\tau_{st} = \tau/(m-1)$" "\n" r"$\xi = m/\tau$",
            color=AXES_CLR, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#cccccc", lw=0.8))

    fig.suptitle("Relaxation Resonance Frequency vs. Bias Current", fontsize=14, y=1.01)
    fig.tight_layout(pad=2.0)
    save(fig, "relaxation_freq_vs_bias.jpg")


fig_10()
