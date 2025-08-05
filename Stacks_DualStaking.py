# app.py  — Dual Stacking Explorer
# streamlit run app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # legend patch for hatched zone

st.set_page_config(page_title="Dual Stacking Explorer", layout="wide")

# -----------------------------------------------------------------------------
# DPI controller stored in session state (used by Figures 1 & 2)
# -----------------------------------------------------------------------------
if "dpi_12" not in st.session_state:
    st.session_state["dpi_12"] = 220  # default high resolution
dpi_12 = st.session_state["dpi_12"]

# ---------------------------- Utilities & formulas ---------------------------
# FIX: no boost on STX-leg; boost only on BTC-leg in Tier-2
def r_stx(phi, T, S0, S2):
    """BTC per STX unit (unboosted STX leg)."""
    return (phi * T) / (S0 + S2)

def r_btc(phi, T, B1, m2, B2):
    """BTC per BTC unit (Tier-2 BTC boosted)."""
    return ((1.0 - phi) * T) / (B1 + m2 * B2)

def apr_base(phi, T, B1, m2, B2):
    """Base is BTC-only in % (BTC terms)."""
    return r_btc(phi, T, B1, m2, B2) * 100.0

def apr_pure(phi, T, S0, S2, rho):
    """Pure staking APR in % (BTC terms) for S0 only."""
    # reward_pure = S0 * r_stx ; capital_pure = S0/rho
    # ⇒ APR_pure = (reward/capital)*100 = r_stx * rho * 100
    return r_stx(phi, T, S0, S2) * rho * 100.0  # NEW

def apr_tier2(phi, T, S0, B1, m2, B2, S2, rho):
    """Tier-2 APR (%) – BTC + STX collateral (STX unboosted, BTC boosted)."""
    rs = r_stx(phi, T, S0, S2)
    rb = r_btc(phi, T, B1, m2, B2)
    reward  = (S2) * rs + (m2 * B2) * rb      # FIX: no m2 on S2
    capital = B2 + (S2 / rho)                 # BTC + STX (BTC terms)
    return (reward / capital) * 100.0

def b2_max_for_base_target(phi, T, B1, m2, base_target_pct):
    """B2_max s.t. APR_base >= base_target_pct."""
    return (((1.0 - phi) * T) / (base_target_pct / 100.0) - B1) / m2

# ----------------------------- Sidebar inputs -------------------------------
st.sidebar.header("Inputs")

# Calibration
phi = st.sidebar.slider(
    "φ — share to STX-leg",
    min_value=0.0, max_value=1.0, value=0.60, step=0.01,
    help="Fraction of total rewards directed to the STX-leg. (1−φ) goes to the BTC-leg."
)
m2  = st.sidebar.slider(
    "m₂ — Tier-2 multiplier",
    min_value=1.0, max_value=6.0, value=3.0, step=0.1,
    help="Boost factor applied **only** to Tier-2 BTC weights."
)
T   = st.sidebar.number_input(
    "T — total rewards (BTC / year)",
    min_value=1.0, value=327.0, step=1.0,
    help="Annual reward budget paid in BTC."
)

# Environment
S0  = st.sidebar.number_input(
    "S₀ — pure STX stakers (STX)",
    min_value=0.0, value=538_800_306.88, step=1e6,
    help="Total STX locked by pure stakers (outside Tier-ladder)."
)
B1  = st.sidebar.number_input(
    "B₁ — Tier-1 BTC deposits (BTC)",
    min_value=0.0, value=5_000.0, step=100.0,
    help="Current BTC deposits on Base tier (Tier-1)."
)
rho = st.sidebar.number_input(
    "ρ — STX per 1 BTC (absolute)",
    min_value=1_000.0, value=float(108_774 / 0.665451), step=5_000.0,
    help="Price ratio used to convert STX to BTC terms in capital."
)

# Tier-2 deposit (for metrics) – choose policy for S2
st.sidebar.subheader("Tier-2 point for metrics")
B2_point = st.sidebar.number_input(
    "B₂ — Tier-2 BTC (BTC)",
    min_value=0.0, value=2_500.0, step=100.0,
    help="BTC deposited in Tier-2 for the point metrics."
)

policy = st.sidebar.radio(
    "Tier-2 collateral policy",
    options=["α × minimum (S₂ = α·10%·B₂·ρ)", "Manual S₂ (STX)"],
    help="How to determine the amount of STX locked in Tier-2."
)

collat_min = st.sidebar.number_input(
    "Minimum collateral (as % of BTC)",
    min_value=0.0, max_value=100.0, value=10.0, step=0.5,
    help="Regulatory/Protocol minimum (e.g., 10% means S₂_min = 0.1·B₂·ρ)."
) / 100.0

if policy.startswith("α"):
    alpha = st.sidebar.number_input(
        "α — over-collateralization factor (≥1)",
        min_value=1.0, value=1.20, step=0.05,
        help="How much STX above the minimum 10% rule (α=1.2 → 12% STX vs BTC)."
    )
    S2_point = alpha * collat_min * B2_point * rho
else:
    S2_point = st.sidebar.number_input(
        "S₂ — Tier-2 STX locked (STX)",
        min_value=0.0, value=0.12 * B2_point * rho, step=1e6,
        help="Absolute STX tokens locked in Tier-2 for the point metrics."
    )
    # effective alpha for display
    alpha = (S2_point / (collat_min * B2_point * rho)) if (B2_point > 0 and rho > 0 and collat_min>0) else np.nan

# Targets
st.sidebar.subheader("Targets")
base_target = st.sidebar.number_input(
    "Base APR target (%)",
    min_value=0.1, value=1.0, step=0.1,
    help="Target APR (BTC terms) required on Tier-1 (Base)."
)
t2_target = st.sidebar.number_input(
    "Tier-2 APR target (%)",
    min_value=0.1, value=2.5, step=0.1,
    help="Target APR (BTC terms) required on Tier-2 (Max)."
)
pure_target = st.sidebar.number_input(   # NEW
    "Pure-staking APR target (%)",
    min_value=0.1, value=1.0, step=0.1,
    help="Target APR (BTC terms) required for pure STX stakers."
)

# ----------------------------- Top metrics -----------------------------------
APR_base_pt = apr_base(phi, T, B1, m2, B2_point)
APR_max_pt  = apr_tier2(phi, T, S0, B1, m2, B2_point, S2_point, rho)
APR_pure_pt = apr_pure(phi, T, S0, S2_point, rho)  # NEW
B2_max_cap  = b2_max_for_base_target(phi, T, B1, m2, base_target)

colA, colB, colC, colD = st.columns(4)
colA.metric("Base APR (Tier-1) — point", f"{APR_base_pt:.2f} %")
with colA.expander(" ?  What is this?"):
    st.write("APR on Tier-1 (BTC-only) at the selected **B₂**; independent of ρ.  \n"
             r"$$\mathrm{APR}_{\text{base}} = \frac{(1-\varphi)T}{B_1+m_2B_2}\times 100.$$")

colB.metric("Tier-2 APR (Max) — point", f"{APR_max_pt:.2f} %")
with colB.expander(" ?  What is this?"):
    st.write("APR on Tier-2 (BTC + STX).  \n"
             r"$$\mathrm{APR}_{\max}=\frac{S_2\frac{\varphi T}{S_0+S_2}+(m_2 B_2)\frac{(1-\varphi)T}{B_1+m_2 B_2}}{B_2+\frac{S_2}{\rho}}\times 100.$$")

colC.metric("Pure-staking APR — point", f"{APR_pure_pt:.2f} %")  # NEW
with colC.expander(" ?  What is this?"):
    st.write("APR for pure STX stakers (S₀ only), in BTC terms.  \n"
             r"$$\mathrm{APR}_{\text{pure}}=\frac{\varphi T}{S_0+S_2}\,\rho \times 100.$$")

colD.metric("Capacity:  B₂^max (Base ≥ target)", f"{B2_max_cap:,.0f} BTC")
with colD.expander(" ?  What is this?"):
    st.write("Maximum Tier-2 BTC that still keeps **Base APR ≥ target**.  \n"
             r"$$B_2^{\max}=\frac{\frac{(1-\varphi)T}{\mathrm{APR}_{\text{base}}^{\ast}/100}-B_1}{m_2}.$$")

st.caption(f"Effective α at the point: {alpha:.2f}× minimum — S₂={S2_point:,.0f} STX")

# ----------------------------- Limit functions -------------------------------
st.markdown("### Cost-opportunity limit functions (targets)")
st.latex(r"""\mathrm{Base:}\quad \frac{(1-\varphi)T}{B_1+m_2 B_2}\times 100=\mathrm{APR}_{\text{base}}^{\ast}""")
st.latex(r"""\mathrm{Tier\text{-}2:}\quad 
\frac{S_2\frac{\varphi T}{S_0+S_2}+(m_2 B_2)\frac{(1-\varphi)T}{B_1+m_2 B_2}}{B_2+\frac{S_2}{\rho}}\times 100
=\mathrm{APR}_{\max}^{\ast}""")  # FIX
st.latex(r"""\mathrm{Pure:}\quad \frac{\varphi T}{S_0+S_2}\,\rho\times 100=\mathrm{APR}_{\text{pure}}^{\ast}""")  # NEW

# ================================ Plots ======================================
st.markdown("---")
st.subheader("1) Calibration map: φ vs m₂  —  color = min(Base APR, Tier-2 APR)")

# Grid φ vs m2
phi_grid = np.linspace(0.0, 1.0, 121)
m2_grid  = np.linspace(1.0, 6.0, 101)
PHI, M2  = np.meshgrid(phi_grid, m2_grid)

# S2 used in this map
if policy.startswith("α"):
    S2_for_map = alpha * collat_min * B2_point * rho
else:
    S2_for_map = S2_point

APR_base_map = ((1.0 - PHI) * T / (B1 + M2 * B2_point)) * 100.0
APR2_map     = np.zeros_like(PHI)
for i in range(APR2_map.shape[0]):
    for j in range(APR2_map.shape[1]):
        APR2_map[i, j] = apr_tier2(PHI[i, j], T, S0, B1, M2[i, j], B2_point, S2_for_map, rho)

# NEW: pure map (horizontal dependence on φ only)
APR_pure_map = apr_pure(PHI, T, S0, S2_for_map, rho)

Z = np.minimum(APR_base_map, APR2_map)

# ---- Figure 1 (higher resolution, same size) --------------------------------
fig1, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=dpi_12)
hm = ax1.pcolormesh(m2_grid, phi_grid, Z.T, shading="auto", cmap="magma")
cb = fig1.colorbar(hm, ax=ax1); cb.set_label("min(APR_base, APR_max) [%]")

# Helper lines (targets)
phi_base_line = 1.0 - (base_target/100.0)*(B1 + m2_grid * B2_point)/T
ax1.plot(m2_grid, phi_base_line, ls="--", color="cyan", label=f"APR₁ = {base_target:.1f}%")

# Tier-2 boundary (nonlinear)
phi_t2_line = []
for m2v in m2_grid:
    vals = np.array([apr_tier2(ph, T, S0, B1, m2v, B2_point, S2_for_map, rho) for ph in phi_grid])
    if (vals.min() <= t2_target <= vals.max()):
        k = np.argmin(np.abs(vals - t2_target))
        phi_t2_line.append((m2v, phi_grid[k]))
phi_t2_line = np.array(phi_t2_line)
if phi_t2_line.size:
    ax1.plot(phi_t2_line[:,0], phi_t2_line[:,1], ls="--", color="white", label=f"APR₂ = {t2_target:.1f}%")

# NEW: pure boundary (horizontal line)
phi_pure_line = (pure_target/100.0) * (S0 + S2_for_map) / (T * (1.0/rho))  # rearranged from APR_pure target
# Simpler: APR_pure = (phi*T*rho*100)/(S0+S2) ⇒ φ* = target*(S0+S2)/(T*rho*100)
phi_pure_line = (pure_target * (S0 + S2_for_map)) / (T * rho * 100.0)
if 0.0 <= phi_pure_line <= 1.0:
    ax1.axhline(phi_pure_line, ls="--", color="magenta", label=f"APR_pure = {pure_target:.1f}%")

# Validity zone (hatched): all three targets
valid1 = (APR_base_map >= base_target) & (APR2_map >= t2_target) & (APR_pure_map >= pure_target)  # NEW
ax1.contourf(m2_grid, phi_grid, valid1.T.astype(float),
             levels=[0.5, 1.1], colors='none', hatches=['////'], alpha=0)

# Legend hatch patch
patch1 = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='////',
                        label='Valid (APR₁, APR₂ & APR_pure ≥ targets)')
handles, labels = ax1.get_legend_handles_labels()
handles.append(patch1); labels.append(patch1.get_label())
ax1.legend(handles, labels, loc="best")

ax1.set_xlabel("m₂ (Max-tier multiplier)")
ax1.set_ylabel("φ (STX-leg share)")
ax1.set_title("Calibration Map: φ vs m₂\nColor = min(APR_base, APR_max)")
st.pyplot(fig1)

# --------------------------- 2) Absolute deposits map ------------------------
st.markdown("---")
st.subheader("2) Coupled APRs with absolute deposits (B₂ vs S₂)")

# Axes
B2_vals = np.linspace(100.0, 10_000.0, 181)
S2_vals = np.linspace(1e6, 250e6, 181)
B2G, S2G = np.meshgrid(B2_vals, S2_vals)

# Valid collateral region (>= minimum)
S2_min = collat_min * B2G * rho
ok = S2G >= S2_min

APR2 = np.full_like(B2G, np.nan, float)
BASE = ((1.0 - phi) * T / (B1 + m2 * B2G)) * 100.0
PURE = apr_pure(phi, T, S0, S2G, rho)  # NEW

for i in range(APR2.shape[0]):
    for j in range(APR2.shape[1]):
        if ok[i, j]:
            APR2[i, j] = apr_tier2(phi, T, S0, B1, m2, B2G[i, j], S2G[i, j], rho)

# ---- Figure 2 (higher resolution, same size) --------------------------------
fig2, ax2 = plt.subplots(figsize=(7.5, 4.8), dpi=dpi_12)
hm = ax2.pcolormesh(B2G, S2G/1e6, APR2, shading="auto")
cb = fig2.colorbar(hm, ax=ax2); cb.set_label("Tier-2 APR (%)")

# White contours: APR targets
if np.nanmin(APR2) <= t2_target <= np.nanmax(APR2):
    cs2 = ax2.contour(B2G, S2G/1e6, APR2, levels=[t2_target], colors="white", linewidths=1.3, linestyles="--")
    ax2.clabel(cs2, inline=True, fmt=lambda v: f"APR₂={v:.1f}%")
if np.nanmin(BASE) <= base_target <= np.nanmax(BASE):
    cs1 = ax2.contour(B2G, S2G/1e6, BASE, levels=[base_target], colors="white", linewidths=1.3, linestyles="--")
    ax2.clabel(cs1, inline=True, fmt=lambda v: f"APR₁={v:.1f}%")

# NEW: magenta line — APR_pure target (horizontal, since depends only on S2 here)
S2_star = (phi*T*rho*100.0)/pure_target - S0
if S2_vals.min() <= S2_star <= S2_vals.max():
    ax2.axhline(S2_star/1e6, color="magenta", linestyle="--", label=f"APR_pure = {pure_target:.1f}%")

# Cyan dashed: min collateral
ax2.plot(B2_vals, (collat_min * B2_vals * rho)/1e6, ls="--", color="cyan", label="Min 10% STX collateral")

# Validity zone (hatched) = collateral OK & all three APR targets
valid2 = ok & (APR2 >= t2_target) & (BASE >= base_target) & (PURE >= pure_target)  # NEW
ax2.contourf(B2G, S2G/1e6, valid2.astype(float),
             levels=[0.5, 1.1], colors='none', hatches=['////'], alpha=0)

# Legend hatch patch
patch2 = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='////',
                        label='Valid (collateral & APR₁/APR₂/APR_pure ≥ targets)')
handles2, labels2 = ax2.get_legend_handles_labels()
handles2.append(patch2); labels2.append(patch2.get_label())
ax2.legend(handles2, labels2, loc="best")

ax2.set_xlabel("Tier-2 BTC deposits (B₂)")
ax2.set_ylabel("Tier-2 STX locked (millions)")
ax2.set_title("Absolute Deposits: B₂ vs S₂\nHeatmap: Tier-2 APR; Contours: APR targets")
st.pyplot(fig2)

# --------------------------- 3) Budget sensitivity (T vs B2) -----------------
st.markdown("---")
st.subheader("3) Budget sensitivity (φ fixed): T vs B₂  — α × min collateral")

# Axes
T_vals  = np.linspace(1.0, 1000.0, 301)
B2_vals = np.linspace(100.0, 6_000.0, 241)
TT, B2GG = np.meshgrid(T_vals, B2_vals)
S2GG = alpha * collat_min * B2GG * rho

APR2B = np.zeros_like(TT)
BASEB = np.zeros_like(TT)
PUREB = apr_pure(phi, TT, S0, S2GG, rho)  # NEW
for i in range(APR2B.shape[0]):
    for j in range(APR2B.shape[1]):
        APR2B[i, j] = apr_tier2(phi, TT[i, j], S0, B1, m2, B2GG[i, j], S2GG[i, j], rho)
        BASEB[i, j] = apr_base(phi, TT[i, j], B1, m2, B2GG[i, j])

fig3, ax3 = plt.subplots(figsize=(7.5, 4.8))
hm = ax3.pcolormesh(T_vals, B2_vals, APR2B, shading="auto")
cb = fig3.colorbar(hm, ax=ax3); cb.set_label("Tier-2 APR (%)")

# Contours (APR targets)
if np.nanmin(APR2B) <= t2_target <= np.nanmax(APR2B):
    cs2 = ax3.contour(T_vals, B2_vals, APR2B, levels=[t2_target], colors="white", linewidths=1.2)
    ax3.clabel(cs2, inline=True, fmt=lambda v: f"APR₂={v:.1f}%")
if np.nanmin(BASEB) <= base_target <= np.nanmax(BASEB):
    cs1 = ax3.contour(T_vals, B2_vals, BASEB, levels=[base_target], colors="white", linestyles="--", linewidths=1.2)
    ax3.clabel(cs1, inline=True, fmt=lambda v: f"APR₁={v:.1f}%")
# NEW: pure-staking contour
if np.nanmin(PUREB) <= pure_target <= np.nanmax(PUREB):
    csp = ax3.contour(T_vals, B2_vals, PUREB, levels=[pure_target], colors="magenta", linestyles="--", linewidths=1.2)
    ax3.clabel(csp, inline=True, fmt=lambda v: f"APR_pure={v:.1f}%")

# Validity zone (hatched): all three
valid3 = (APR2B >= t2_target) & (BASEB >= base_target) & (PUREB >= pure_target)
ax3.contourf(T_vals, B2_vals, valid3.astype(float),
             levels=[0.5, 1.1], colors='none', hatches=['////'], alpha=0)

# Legend hatch patch
patch3 = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='////',
                        label='Valid (APR₁, APR₂ & APR_pure ≥ targets)')
handles3, labels3 = ax3.get_legend_handles_labels()
handles3.append(patch3); labels3.append(patch3.get_label())
ax3.legend(handles3, labels3, loc="best")

ax3.set_xlabel("Total reward T (BTC / year)")
ax3.set_ylabel("Tier-2 BTC deposits (B₂)")
ax3.set_title(f"Budget Sensitivity (φ fixed={phi:.2f}): Tier-2 APR vs T and B₂\nα={alpha:.2f}× min collateral | B₁={B1:.0f} | m₂={m2:.1f}")
st.pyplot(fig3)

# --------------------------- 4) ρ vs B2 (APR2-only zone) ---------------------
st.markdown("---")
st.subheader("4) Tier-2 APR vs absolute ρ and B₂  —  APR₂≥target hatched (pure not enforced)")

rho_vals = np.linspace(50_000.0, 650_000.0, 321)
RHO, B2H = np.meshgrid(rho_vals, np.linspace(100.0, 8_000.0, 241))
S2H = alpha * collat_min * B2H * RHO

APR2H = np.zeros_like(RHO)
for i in range(APR2H.shape[0]):
    for j in range(APR2H.shape[1]):
        APR2H[i, j] = apr_tier2(phi, T, S0, B1, m2, B2H[i, j], S2H[i, j], RHO[i, j])

valid_apr2 = APR2H >= t2_target

fig4, ax4 = plt.subplots(figsize=(7.5, 4.8))
hm = ax4.pcolormesh(rho_vals, B2H[:,0], APR2H, shading="auto")
cb = fig4.colorbar(hm, ax=ax4); cb.set_label("Tier-2 APR (%)")
if np.nanmin(APR2H) <= t2_target <= np.nanmax(APR2H):
    cs2 = ax4.contour(rho_vals, B2H[:,0], APR2H, levels=[t2_target], colors="white", linewidths=1.2)
    ax4.clabel(cs2, inline=True, fmt=lambda v: f"APR₂={v:.1f}%")
ax4.contourf(rho_vals, B2H[:,0], valid_apr2.astype(float),
             levels=[0.5, 1.1], colors='none', hatches=['////'], alpha=0)
ax4.set_xlabel("ρ (STX per 1 BTC) — absolute")
ax4.set_ylabel("Tier-2 BTC deposits (B₂)")
ax4.set_title(f"Tier-2 APR vs ρ and B₂  (α={alpha:.2f}× min collateral, φ={phi:.2f}, m₂={m2:.1f})")
st.pyplot(fig4)

# -----------------------------------------------------------------------------
# DPI controller placed at the END (affects Figures 1 & 2 on next redraw)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Display settings")
st.slider(
    "DPI for Figures 1 & 2 (resolution only, size unchanged)",
    min_value=120, max_value=360, step=10, key="dpi_12",
    help="Higher DPI = sharper images. Move the slider to refresh figures 1 & 2."
)
st.caption("Tip: after adjusting DPI, the app reruns and the new setting is applied to Figures 1 & 2.")
