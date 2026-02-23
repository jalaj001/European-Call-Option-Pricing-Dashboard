import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Pricing Models (UNCHANGED)
# =========================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call(S0, K, r, sigma, T):
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        ST = S0 * math.exp(r * T)
        return math.exp(-r * T) * max(ST - K, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def binomial_crr_call(S0, K, r, sigma, T, N):
    if N < 1:
        return black_scholes_call(S0, K, r, sigma, T)

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    j = np.arange(N + 1)
    ST = S0 * (u ** j) * (d ** (N - j))
    payoff = np.maximum(ST - K, 0.0)

    for _ in range(N):
        payoff = disc * (p * payoff[1:] + (1 - p) * payoff[:-1])

    return float(payoff[0])

def mc_call_price(S0, K, r, sigma, T, M, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(M)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Quant Option Pricing Dashboard", layout="wide")

st.markdown("## 📈 Quant Option Pricing Dashboard")
st.caption("Black–Scholes · Monte Carlo · Binomial Tree")

# -------------------------
# Input Panel
# -------------------------

st.markdown("### Input Parameters")
c1, c2, c3, c4 = st.columns(4)

with c1:
    S0 = st.number_input("Stock Price (S)", value=100.0)
with c2:
    K = st.number_input("Strike Price (K)", value=100.0)
with c3:
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
with c4:
    sigma = st.number_input("Volatility (σ)", value=0.20)

c5, c6, c7 = st.columns(3)
with c5:
    T = st.number_input("Time to Maturity (T)", value=1.0)
with c6:
    N_tree = st.slider("Binomial Steps", 10, 1000, 200)
with c7:
    M = st.slider("Monte Carlo Simulations", 1000, 1000000, 50000)

st.divider()

# -------------------------
# Pricing
# -------------------------

bs_price = black_scholes_call(S0, K, r, sigma, T)
bt_price = binomial_crr_call(S0, K, r, sigma, T, N_tree)
mc_price = mc_call_price(S0, K, r, sigma, T, M)

# -------------------------
# KPI Cards
# -------------------------

k1, k2, k3 = st.columns(3)

k1.metric("🟩 Black-Scholes", f"{bs_price:.4f}")
k2.metric("🟦 Monte Carlo", f"{mc_price:.4f}")
k3.metric("🟨 Binomial Tree", f"{bt_price:.4f}")

st.divider()

# -------------------------
# Model Comparison Chart
# -------------------------

# st.markdown("### Model Comparison")

# df = pd.DataFrame({
#     "Model": ["Black-Scholes", "Monte Carlo", "Binomial Tree"],
#     "Price": [bs_price, mc_price, bt_price]
# })

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.bar(df["Model"], df["Price"])
# ax.set_ylabel("Option Price")
# ax.set_title("European Call Option Pricing Comparison")
# st.pyplot(fig)

st.markdown("### Model Comparison")

df = pd.DataFrame({
    "Model": ["Black-Scholes", "Monte Carlo", "Binomial Tree"],
    "Price": [bs_price, mc_price, bt_price]
})

fig, ax = plt.subplots(figsize=(6, 3))  # smaller + responsive
ax.bar(df["Model"], df["Price"])
ax.set_ylabel("Option Price")
ax.set_title("European Call Option Pricing")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()   # 🔑 prevents clipping
st.pyplot(fig, width='stretch')

# -------------------------
# Footer
# -------------------------

st.caption("📊 Quantitative Finance Dashboard · Streamlit")

st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #0e2a47;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #0b2239;
    }

    /* Text color */
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    /* Input boxes */
    input, select, textarea {
        background-color: #123a63 !important;
        color: white !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #123a63;
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
