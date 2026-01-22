import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import streamlit as st

# =========================================================
# Helpers
# =========================================================
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def eu_to_pct(x_eu, lrv, urv):
    span = (urv - lrv)
    if abs(span) < 1e-12:
        span = 1.0
    return 100.0 * (x_eu - lrv) / span

# =========================================================
# Deadtime (minutes)
# =========================================================
class DeadTimeMin:
    def __init__(self, dead_min: float, dt_min: float, init: float = 0.0):
        dt_min = max(dt_min, 1e-12)
        self.n = int(round(max(dead_min, 0.0) / dt_min))
        self.buf = deque([init] * self.n, maxlen=self.n) if self.n > 0 else None

    def step(self, u: float) -> float:
        if self.n <= 0:
            return u
        self.buf.append(u)
        return self.buf[0]

# =========================================================
# PT1 (Aspen/dmcplus) in MINUTES
# gain is EU/%OP
# =========================================================
class PT1_Aspen_Min:
    def __init__(self, gain: float, tau_min: float, dead_min: float, dt_min: float,
                 y0: float = 0.0, u0: float = 0.0):
        self.K = float(gain)
        self.tau = max(float(tau_min), 1e-12)
        self.dt = float(dt_min)
        self.y = float(y0)
        self.delay = DeadTimeMin(dead_min, dt_min, init=u0)

    def step(self, u: float) -> float:
        u_d = self.delay.step(float(u))
        dy = (-self.y + self.K * u_d) / self.tau
        self.y += dy * self.dt
        return self.y

# =========================================================
# Honeywell EqA..EqE PID in MINUTES, PV% domain
# Output is delta OP% around 0; OP0 handled outside.
# =========================================================
class HoneywellPID_Min_PVpct:
    def __init__(self, eq: str, Kc: float, Ti_min: float, Td_min: float, dt_min: float,
                 reverse: bool = False, d_filter_min: float = 0.0):
        self.eq = eq
        self.Kc = -float(Kc) if reverse else float(Kc)
        self.Ti = max(float(Ti_min), 0.0)
        self.Td = max(float(Td_min), 0.0)
        self.dt = max(float(dt_min), 1e-12)
        self.d_filter = max(float(d_filter_min), 0.0)

        self.i = 0.0
        self.prev_e = 0.0
        self.prev_pv = 0.0
        self.d_f = 0.0

    def _p_sig(self, e, pv_pct):
        if self.eq in ("EqA", "EqB", "EqE"):
            return e
        if self.eq == "EqC":
            return -pv_pct
        if self.eq == "EqD":
            return 0.0
        return e

    def step(self, sp_pct: float, pv_pct: float) -> float:
        e = float(sp_pct - pv_pct)

        # Determine structure
        if self.eq == "EqA":
            use_i = True; d_on_pv = False; use_d = self.Td > 1e-12
        elif self.eq == "EqB":
            use_i = True; d_on_pv = True;  use_d = self.Td > 1e-12
        elif self.eq == "EqC":
            use_i = True; d_on_pv = True;  use_d = self.Td > 1e-12
        elif self.eq == "EqD":
            use_i = True; d_on_pv = False; use_d = False
        elif self.eq == "EqE":
            use_i = False; d_on_pv = False; use_d = False
        else:
            use_i = True; d_on_pv = True;  use_d = self.Td > 1e-12

        p_sig = self._p_sig(e, pv_pct)
        up = self.Kc * p_sig

        ui = 0.0
        if use_i and self.Ti > 1e-12:
            self.i += (self.Kc / self.Ti) * e * self.dt
            ui = self.i

        ud = 0.0
        if use_d:
            if d_on_pv:
                raw = -self.Kc * self.Td * (pv_pct - self.prev_pv) / self.dt
            else:
                raw =  self.Kc * self.Td * (e - self.prev_e) / self.dt

            if self.d_filter > 1e-12:
                a = self.dt / (self.d_filter + self.dt)
                self.d_f += a * (raw - self.d_f)
                ud = self.d_f
            else:
                ud = raw

        self.prev_e = e
        self.prev_pv = float(pv_pct)

        return up + ui + ud

    def bumpless_init(self, sp_pct: float, pv_pct: float):
        e0 = float(sp_pct - pv_pct)
        self.prev_e = e0
        self.prev_pv = float(pv_pct)
        self.d_f = 0.0

        # Determine if I is used
        if self.eq == "EqE":
            return
        if self.eq == "EqD":
            self.i = 0.0
            return

        # set i so that Kc*p_sig + i ≈ 0 at start (delta≈0)
        p_sig0 = self._p_sig(e0, pv_pct)
        if self.Ti > 1e-12:
            self.i = -self.Kc * p_sig0
        else:
            self.i = 0.0

# =========================================================
# Simulation: PT1 + SP step
# OP(t) = clamp(OP0 + PID_delta, OPmin, OPmax)
# =========================================================
def simulate_pt1_sp_step(
    gain_eu_per_op: float, tau_min: float, dead_min: float,
    pv_lrv: float, pv_urv: float,
    op_min: float, op_max: float,
    eq: str, Kc: float, Ti_min: float, Td_min: float, reverse: bool, d_filter_min: float,
    dt_min: float, t_end_min: float,
    sp0_eu: float, sp_step_eu: float, t_step_min: float,
    op0_pct: float
):
    dt_min = max(float(dt_min), 1e-12)
    n = int(t_end_min / dt_min) + 1
    t = np.linspace(0, t_end_min, n)

    sp = np.ones(n) * float(sp0_eu)
    sp[t >= float(t_step_min)] = float(sp0_eu) + float(sp_step_eu)

    pv0 = float(sp0_eu)
    op0 = clamp(float(op0_pct), float(op_min), float(op_max))

    proc = PT1_Aspen_Min(gain=gain_eu_per_op, tau_min=tau_min, dead_min=dead_min,
                         dt_min=dt_min, y0=pv0, u0=op0)

    pid = HoneywellPID_Min_PVpct(eq=eq, Kc=Kc, Ti_min=Ti_min, Td_min=Td_min, dt_min=dt_min,
                                 reverse=reverse, d_filter_min=d_filter_min)

    sp0_pct = eu_to_pct(sp0_eu, pv_lrv, pv_urv)
    pv0_pct = eu_to_pct(pv0,    pv_lrv, pv_urv)
    pid.bumpless_init(sp0_pct, pv0_pct)

    pv = np.zeros(n)
    op = np.zeros(n)
    pv[0] = pv0
    op[0] = op0

    for k in range(1, n):
        sp_pct = eu_to_pct(sp[k], pv_lrv, pv_urv)
        pv_pct = eu_to_pct(pv[k-1], pv_lrv, pv_urv)

        du = pid.step(sp_pct, pv_pct)  # delta OP%
        op_cmd = op0 + du
        op[k] = clamp(op_cmd, op_min, op_max)

        pv[k] = proc.step(op[k])

    return t, sp, pv, op

# =========================================================
# Plot: SP/PV left axis, OP right axis with auto-zoom
# =========================================================
def make_plot(t, sp, pv, op, title, op_min, op_max):
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=160)

    # SP/PV (EU)
    ax.step(t, sp, where="post", label="SP [EU]", linewidth=2.0)
    ax.plot(t, pv, label="PV [EU]", linewidth=3.0)  # PV thicker
    ax.set_xlabel("t [min]")
    ax.set_ylabel("PV / SP [EU]")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Zoom EU axis
    y_min = float(min(np.min(sp), np.min(pv)))
    y_max = float(max(np.max(sp), np.max(pv)))
    pad = max(0.5, 0.08 * (y_max - y_min + 1e-9))
    ax.set_ylim(y_min - pad, y_max + pad)

    # OP (%), green, auto-scale
    ax2 = ax.twinx()
    ax2.plot(t, op, label="OP [%]", linewidth=2.0, color="green")
    ax2.set_ylabel("OP [%]")

    op_lo = float(np.min(op)); op_hi = float(np.max(op))
    span = max(1e-6, op_hi - op_lo)
    pad2 = max(1.0, 0.10 * span)
    lo = max(float(op_min), op_lo - pad2)
    hi = min(float(op_max), op_hi + pad2)
    if hi - lo < 2.0:
        mid = 0.5 * (hi + lo)
        lo = max(float(op_min), mid - 1.5)
        hi = min(float(op_max), mid + 1.5)
    ax2.set_ylim(lo, hi)

    # Legend combined
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")

    return fig

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(layout="wide")
st.title("PID Tuning Simulator (Minimal) – PT1 + SP-Step")

with st.sidebar:
    st.header("Prozess (PT1)")
    gain = st.number_input("Gain (EU/%OP)", value=1.954, format="%.6f")
    tau  = st.number_input("Tau (min)", value=6.0, format="%.6f")
    dead = st.number_input("Deadtime (min)", value=1.0, format="%.6f")

    st.header("Ranges")
    pv_lrv = st.number_input("PV LRV (EU)", value=0.0, format="%.6f")
    pv_urv = st.number_input("PV URV (EU)", value=200.0, format="%.6f")
    op_min = st.number_input("OP Min (%)", value=0.0, format="%.3f")
    op_max = st.number_input("OP Max (%)", value=100.0, format="%.3f")

    st.header("SP-Step (Servo)")
    sp0     = st.number_input("SP0 (EU)", value=102.0, format="%.6f")
    sp_step = st.number_input("SP Step (EU)", value=1.0, format="%.6f")
    t_step  = st.number_input("t_step (min)", value=1.0, format="%.6f")

    st.header("Arbeitspunkt")
    op0 = st.number_input("OP0 (%)", value=52.0, format="%.6f")

    st.header("PID (Honeywell)")
    eq = st.selectbox("Equation", ["EqA", "EqB", "EqC", "EqD", "EqE"], index=1)
    reverse = st.checkbox("Reverse", value=False)

    Kc = st.number_input("Kc", value=0.5, format="%.6f")
    Ti = st.number_input("Ti (min)", value=12.0, format="%.6f")
    Td = st.number_input("Td (min)", value=0.0, format="%.6f")
    d_filter = st.number_input("D-Filter (min)", value=0.0, format="%.6f")

    st.header("Simulation")
    dt = st.number_input("dt (min)", value=0.01, format="%.6f")
    t_end = st.number_input("t_end (min)", value=30.0, format="%.3f")

    run = st.button("Simulieren", type="primary")

col1, col2 = st.columns([2, 1], gap="large")
with col2:
    st.subheader("Hinweise")
    st.write(
        "- Minimal-Version: **PT1 + SP-Step**\n"
        "- OP ist **Bias OP0 + ΔOP** aus PID\n"
        "- PV startet bei **SP0** (Arbeitsannahme)\n"
        "- Nächster Schritt: **Vorher/Nachher**, **PT2**, **IMC Button**, **KPIs**"
    )

with col1:
    if run:
        # basic input sanity
        if abs(pv_urv - pv_lrv) < 1e-9:
            st.error("PV URV und LRV dürfen nicht gleich sein.")
        elif dt <= 0 or t_end <= 0:
            st.error("dt und t_end müssen > 0 sein.")
        else:
            t, sp, pv, op = simulate_pt1_sp_step(
                gain_eu_per_op=gain, tau_min=tau, dead_min=dead,
                pv_lrv=pv_lrv, pv_urv=pv_urv,
                op_min=op_min, op_max=op_max,
                eq=eq, Kc=Kc, Ti_min=Ti, Td_min=Td, reverse=reverse, d_filter_min=d_filter,
                dt_min=dt, t_end_min=t_end,
                sp0_eu=sp0, sp_step_eu=sp_step, t_step_min=t_step,
                op0_pct=op0
            )
            fig = make_plot(t, sp, pv, op, "PT1 + SP-Step", op_min, op_max)
            st.pyplot(fig, clear_figure=True)
    else:
        st.info("Links Parameter einstellen und **Simulieren** drücken.")
