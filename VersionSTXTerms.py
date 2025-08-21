# app_churn.py — Dual Stacking Explorer (full app + churn dynamics + fixed axes)
# Run: streamlit run app_churn.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # legend patch for hatched zone

st.set_page_config(page_title="Dual Stacking Explorer (with churn)", layout="wide")

# ---------------------------------------------------------------------------
# DPI controller stored in session state (used by Figures 1–4)
# ---------------------------------------------------------------------------
if "dpi_12" not in st.session_state:
    st.session_state["dpi_12"] = 220  # default high resolution
dpi_12 = st.session_state["dpi_12"]

# ---------------------------- Utilities & formulas --------------------------
def r_stx(phi, T_eff, S0, S2):
    """BTC per STX unit (unboosted STX leg)."""
    return (phi * T_eff) / (S0 + S2)

def r_btc(phi, T_eff, B1, m2, B2):
    """BTC per BTC unit (Tier-2 BTC boosted with m2 in the denominator)."""
    return ((1.0 - phi) * T_eff) / (B1 + m2 * B2)

def apr_base(phi, T_eff, B1, m2, B2):
    """Base APR (Tier-1) in BTC terms (%)."""
    return r_btc(phi, T_eff, B1, m2, B2) * 100.0

def apr_pure(phi, T_eff, S0, S2_eff, rho):
    """Pure staking APR (BTC terms, %) for S0 only."""
    return r_stx(phi, T_eff, S0, S2_eff) * rho * 100.0

def apr_tier2_composite(phi, T_eff, S0, B1, m2, B2, S2_eff, rho, exclude_s2_from_capital: bool):
    """Tier-2 APR (%) — BTC + STX capital; STX unboosted, BTC boosted."""
    rs = r_stx(phi, T_eff, S0, S2_eff)         # BTC per STX
    rb = r_btc(phi, T_eff, B1, m2, B2)         # BTC per BTC
    reward  = (S2_eff) * rs + (m2 * B2) * rb
    capital = B2 + (0.0 if exclude_s2_from_capital else (S2_eff / rho))
    return (reward / capital) * 100.0 if capital > 0 else 0.0

def apr_tier2_btc_only(phi, T_eff, B1, m2, B2):
    """APR (%) on Tier-2 BTC capital only (excl. STX capital)."""
    return m2 * r_btc(phi, T_eff, B1, m2, B2) * 100.0

def b2_max_for_base_target(phi, T_eff, B1, m2, base_target_pct):
    """B2_max s.t. APR_base >= base_target_pct."""
    return (((1.0 - phi) * T_eff) / (base_target_pct / 100.0) - B1) / m2

# ----------------------------- Sidebar (clean layout) ------------------------
# 1) Reward Policy
st.sidebar.header("⚙️ Reward Policy")
phi = st.sidebar.slider(
    "φ — Share to STX leg",
    min_value=0.0, max_value=1.0, value=0.60, step=0.01,
    help="Fraction of the total reward budget allocated to the STX leg. (1−φ) goes to the BTC leg."
)
T_input = st.sidebar.number_input(
    "T — Total rewards (BTC / year)",
    min_value=0.0, value=327.0, step=1.0,
    help="Annual reward budget in BTC (aggregate across both legs). Can be overridden by the modeled value below."
)
m2  = st.sidebar.slider(
    "m₂ — Tier-2 multiplier",
    min_value=1.0, max_value=6.0, value=3.0, step=0.1,
    help="Boost factor applied ONLY to Tier-2 BTC weights in reward allocation."
)

# Rewards model (exponential ×26)
with st.sidebar.expander("Rewards model (optional)", expanded=False):
    st.markdown(
        "Modeled form (annualized with **×26 cycles/year**):  \n"
        r"**Rewards Calculated (BTC)** = \( 26 \cdot a \cdot e^{\,b\cdot(\rho / \rho_{\max})} \)  \n"
        "where ρ is your current **STX per BTC**."
    )
    a_param = st.number_input(
        "a (BTC per cycle)",
        min_value=0.0, value=61.99, step=0.1,
        help="Scale parameter of the per-cycle model; multiplied by 26 to annualize."
    )
    b_param = st.number_input(
        "b (unitless)",
        min_value=-10.0, max_value=0.0, value=-2.345, step=0.001,
        help="Exponent slope; typically negative if rewards shrink as ρ increases."
    )
    rho_norm_max = st.number_input(
        "ρ_max (normalization)",
        min_value=1_000.0, value=160_000.0, step=1_000.0,
        help="Normalization constant to scale ρ inside the exponent (use max ρ of your dataset)."
    )
use_modeled_T = st.sidebar.checkbox(
    "Use modeled rewards instead of T",
    value=False,
    help="If enabled, the model's 'Rewards Calculated (BTC)' replaces T in all computations."
)

# 2) Environment
st.sidebar.header("🌍 Environment")
S0  = st.sidebar.number_input(
    "S₀ — Pure STX stakers (STX)",
    min_value=0.0, value=538_800_306.88, step=1e6,
    help="Total STX locked by pure stakers (outside the Tier-2 ladder)."
)
rho = st.sidebar.number_input(
    "ρ — STX per 1 BTC (absolute)",
    min_value=1_000.0, value=float(108_774 / 0.665451), step=5_000.0,
    help="Price ratio used to convert STX to BTC terms in capital. Higher ρ ⇒ cheaper STX vs BTC."
)

# 3) Tier-1 (Base) inputs
st.sidebar.header("🏛️ Tier-1 (Base)")
B1  = st.sidebar.number_input(
    "B₁ — Tier-1 BTC deposits (BTC)",
    min_value=0.0, value=5_000.0, step=100.0,
    help="Current BTC deposits on Base tier (Tier-1)."
)

# 4) Tier-2 (Max) policy & inputs
st.sidebar.header("🏦 Tier-2 (Max)")
B2_point = st.sidebar.number_input(
    "B₂ — Tier-2 BTC (BTC)",
    min_value=0.0, value=2_500.0, step=100.0,
    help="BTC deposited in Tier-2 for point metrics and charts."
)
policy = st.sidebar.radio(
    "Tier-2 collateral policy",
    options=["α × minimum (S₂ = α·10%·B₂·ρ)", "Manual S₂ (STX)"],
    help="How to determine the amount of STX locked in Tier-2 at the evaluation point."
)
collat_min = st.sidebar.number_input(
    "Minimum collateral (as % of BTC)",
    min_value=0.0, max_value=100.0, value=10.0, step=0.5,
    help="Protocol minimum (e.g., 10% ⇒ S₂_min = 0.1·B₂·ρ)."
) / 100.0
if policy.startswith("α"):
    alpha = st.sidebar.number_input(
        "α — Over-collateralization factor (≥1)",
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
    alpha = (S2_point / (collat_min * B2_point * rho)) if (B2_point > 0 and rho > 0 and collat_min>0) else np.nan

# 5) Targets
st.sidebar.header("🎯 Targets")
base_target = st.sidebar.number_input(
    "Base APR target (%)", min_value=0.1, value=1.0, step=0.1,
    help="Target APR (BTC terms) required on Tier-1 (Base)."
)
t2_target = st.sidebar.number_input(
    "Tier-2 APR target (%)", min_value=0.1, value=2.5, step=0.1,
    help="Target APR (BTC terms) required on Tier-2 (Composite)."
)
pure_target = st.sidebar.number_input(
    "Pure-staking APR target (%)", min_value=0.1, value=1.0, step=0.1,
    help="Target APR (BTC terms) required for pure STX stakers."
)

# 6) Global APR behaviour options
st.sidebar.header("🧰 Display / Behavior")
exclude_s2_toggle = st.sidebar.checkbox(
    "Exclude Tier-2 STX collateral from APRs",
    value=False,
    help="If enabled, APR calculations ignore S₂ both in STX-leg dilution and in capital denominators. "
         "Useful to see APRs without collateral-induced dilution. Rewards/gains remain based on true pool rates."
)

# 7) Churn model (optional)
st.sidebar.header("🔄 Churn Model (optional)")
enable_churn = st.sidebar.checkbox(
    "Enable churn dynamics",
    value=False,
    help="If enabled, BTC deposits flow between Tier-1 and Tier-2 based on APR incentive and collateral pressure."
)
lambda_sens = st.sidebar.slider(
    "λ — Sensitivity to APR difference",
    0.0, 1.0, 0.20, 0.01,
    help="How strongly BTC migrates when APR₂ − APR₁ is large. Higher λ ⇒ stronger response."
)
gamma_comfort = st.sidebar.slider(
    "γ — Comfort collateral ratio",
    0.0, 5.0, 2.00, 0.10,
    help="Comfort threshold for collateral adequacy: γ = S₂ / (ρ·B₂). Above this, Tier-2 becomes less attractive."
)
kappa_penalty = st.sidebar.slider(
    "κ — Over-collateral penalty",
    0.0, 1.0, 0.10, 0.01,
    help="Penalty factor applied when the collateral ratio exceeds the comfort level γ."
)
delta_max = st.sidebar.slider(
    "δₘₐₓ — Max migration per step (%)",
    0.0, 0.5, 0.05, 0.01,
    help="Maximum fraction of total BTC (B₁+B₂) that can migrate per churn step."
)
n_steps = st.sidebar.slider(
    "n_steps — Iterations",
    1, 50, 20, 1,
    help="Number of churn iterations to approximate a quasi-equilibrium."
)

# -------------------- Rewards Calculated (exponential ×26, non-negative) -----
rho_scaled = np.clip(rho / max(rho_norm_max, 1e-9), 0.0, 10.0)
T_modeled_raw = 26.0 * a_param * np.exp(b_param * rho_scaled)
T_modeled = float(np.maximum(T_modeled_raw, 0.0))  # >= 0 to keep plots sane

st.sidebar.metric(
    "Rewards Calculated (BTC)",
    f"{T_modeled:,.3f}",
    help="Annual modeled rewards at the current ρ (26 cycles/year)."
)

# Effective T used throughout the app
T_eff = float(T_modeled) if use_modeled_T else float(T_input)
T_label = f"{T_eff:.0f} BTC/yr" + (" (modeled×26)" if use_modeled_T else "")

# --------------------------- Apply churn (multi-steps) -----------------------
B1_eff, B2_eff, S2_eff_point = float(B1), float(B2_point), float(S2_point)
churn_net_flow = 0.0  # +BTC moved from B1→B2

if enable_churn:
    for _ in range(n_steps):
        S2_eff_for_apr = 0.0 if exclude_s2_toggle else S2_eff_point
        apr1 = apr_base(phi, T_eff, B1_eff, m2, B2_eff)
        apr2 = apr_tier2_composite(phi, T_eff, S0, B1_eff, m2, B2_eff, S2_eff_for_apr, rho, exclude_s2_toggle)
        collat_ratio = (S2_eff_point / (rho * B2_eff)) if B2_eff > 0 else 0.0
        flow = lambda_sens * (apr2 - apr1) / (abs(apr1) + 1e-9)
        if collat_ratio > gamma_comfort:
            flow -= kappa_penalty * (collat_ratio - gamma_comfort)
        flow = float(np.clip(flow, -delta_max, delta_max))
        total_btc = B1_eff + B2_eff
        move = flow * total_btc  # BTC moved from Tier-1 to Tier-2
        B1_eff = max(B1_eff - move, 0.0)
        B2_eff = max(B2_eff + move, 0.0)
        churn_net_flow += move

# ----------------------------- Top metrics (post-churn) ----------------------
S2_eff_for_apr_point = 0.0 if exclude_s2_toggle else S2_eff_point

APR_base_pt      = apr_base(phi, T_eff, B1_eff, m2, B2_eff)
APR_t2_comp      = apr_tier2_composite(phi, T_eff, S0, B1_eff, m2, B2_eff, S2_eff_for_apr_point, rho, exclude_s2_toggle)
APR_t2_btc       = apr_tier2_btc_only(phi, T_eff, B1_eff, m2, B2_eff)
APR_pure_pt      = apr_pure(phi, T_eff, S0, S2_eff_for_apr_point, rho)
B2_max_cap       = b2_max_for_base_target(phi, T_eff, B1_eff, m2, base_target)

colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Base APR (Tier-1) — point", f"{APR_base_pt:.2f} %")
with colA.expander(" ?  What is this?"):
    st.write(
        "APR on Tier-1 (BTC only); independent of ρ.  \n"
        r"$$\mathrm{APR}_{\text{base}}=\frac{(1-\varphi)T_{\text{eff}}}{B_1^{\ast}+m_2B_2^{\ast}}\times100.$$"
    )
colB.metric("Tier-2 APR — composite", f"{APR_t2_comp:.2f} %")
with colB.expander(" ?  What is this?"):
    st.write(
        "APR on Tier-2 including BTC & STX capital (STX unboosted, BTC boosted).  \n"
        "If **Exclude S₂** is enabled, the APR ignores S₂ in both dilution and capital denominator.  \n"
        r"$$\mathrm{APR}_{\max}=\frac{S_2^{(\mathrm{eff})}\frac{\varphi T_{\text{eff}}}{S_0+S_2^{(\mathrm{eff})}}+(m_2B_2^{\ast})\frac{(1-\varphi)T_{\text{eff}}}{B_1^{\ast}+m_2 B_2^{\ast}}}{B_2^{\ast}+\mathbf{1}_{\neg \mathrm{excl}}\,\frac{S_2^{(\mathrm{eff})}}{\rho}}\times100.$$"
    )
colC.metric("Tier-2 APR — BTC-only", f"{APR_t2_btc:.2f} %")
with colC.expander(" ?  What is this?"):
    st.write(
        "APR on Tier-2 BTC **only** (excl. STX capital).  \n"
        r"$$\mathrm{APR}^{(\mathrm{BTC})}_{2}=m_2\,\frac{(1-\varphi)T_{\text{eff}}}{B_1^{\ast}+m_2B_2^{\ast}}\times100.$$"
    )
colD.metric("Pure-staking APR — point (BTC terms)", f"{APR_pure_pt:.2f} %")
with colD.expander(" ?  What is this?"):
    st.write("APR for pure STX stakers (S₀ only). Toggle can exclude S₂ from dilution.")
    st.latex(r"""\mathrm{APR}_{\text{pure}}^{(\mathrm{BTC\ terms})}
    =\frac{\varphi T_{\text{eff}}}{S_0+S_2^{(\mathrm{eff})}}\;\rho\times100""")
colE.metric("Capacity:  B₂^max (Base ≥ target)", f"{B2_max_cap:,.0f} BTC")
with colE.expander(" ?  What is this?"):
    st.write(
        "Maximum Tier-2 BTC that still keeps **Base APR ≥ target** (using post-churn B₁*, B₂*).  \n"
        r"$$B_2^{\max}=\frac{\frac{(1-\varphi)T_{\text{eff}}}{\mathrm{APR}_{\text{base}}^{\ast}/100}-B_1^{\ast}}{m_2}.$$"
    )

st.caption(
    f"Post-churn state: B₁*={B1_eff:,.0f} BTC | B₂*={B2_eff:,.0f} BTC | "
    f"S₂(actual)={S2_eff_point:,.0f} STX | S₂(eff for APR)={'0' if exclude_s2_toggle else f'{S2_eff_for_apr_point:,.0f}'} STX | "
    f"T_used = {T_label} | Net churn (B₁→B₂) = {churn_net_flow:,.2f} BTC | α={alpha:.2f}× min collateral"
)

# ----------------------------- Limit functions -------------------------------
st.markdown("### Cost-opportunity limit functions (targets)")
st.latex(r"""\mathrm{Base:}\quad \frac{(1-\varphi)T_{\text{eff}}}{B_1^{\ast}+m_2 B_2^{\ast}}\times 100=\mathrm{APR}_{\text{base}}^{\ast}""")
st.latex(r"""\mathrm{Tier\text{-}2\ composite:}\quad 
\frac{S_2^{(\mathrm{eff})}\frac{\varphi T_{\text{eff}}}{S_0+S_2^{(\mathrm{eff})}}+(m_2 B_2^{\ast})\frac{(1-\varphi)T_{\text{eff}}}{B_1^{\ast}+m_2 B_2^{\ast}}}{B_2^{\ast}+\mathbf{1}_{\neg \mathrm{excl}}\,\frac{S_2^{(\mathrm{eff})}}{\rho}}\times 100
=\mathrm{APR}_{\max}^{\ast}""")
st.latex(r"""\mathrm{Tier\text{-}2\ BTC\ only:}\quad 
m_2\,\frac{(1-\varphi)T_{\text{eff}}}{B_1^{\ast}+m_2 B_2^{\ast}}\times 100=\mathrm{APR}_{2}^{(\mathrm{BTC})\ast}""")
st.latex(r"""\mathrm{Pure:}\quad \frac{\varphi T_{\text{eff}}}{S_0+S_2^{(\mathrm{eff})}}\,\rho\times 100=\mathrm{APR}_{\text{pure}}^{\ast}""")

# ================================= Plots =====================================
st.markdown("---")
st.subheader("1) Calibration map: φ vs m₂  —  color = min(Base APR, Tier-2 APR)")

phi_grid = np.linspace(0.0, 1.0, 121)
m2_grid  = np.linspace(1.0, 6.0, 101)
PHI, M2  = np.meshgrid(phi_grid, m2_grid)
S2_for_map_eff = 0.0 if exclude_s2_toggle else S2_eff_point

APR_base_map = ((1.0 - PHI) * T_eff / (B1_eff + M2 * B2_eff)) * 100.0
APR2_map     = np.zeros_like(PHI)
for i in range(APR2_map.shape[0]):
    for j in range(APR2_map.shape[1]):
        APR2_map[i, j] = apr_tier2_composite(PHI[i, j], T_eff, S0, B1_eff, M2[i, j], B2_eff, S2_for_map_eff, rho, exclude_s2_toggle)
APR_pure_map = apr_pure(PHI, T_eff, S0, S2_for_map_eff, rho)
Z = np.minimum(APR_base_map, APR2_map)

fig1, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=dpi_12)
hm = ax1.pcolormesh(m2_grid, phi_grid, Z.T, shading="auto", cmap="magma")
