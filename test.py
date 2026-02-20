import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call(S0, K, r, sigma, T):
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        ST = S0 * math.exp(r * T)
        return math.exp(-r * T) * max(ST - K, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def binomial_crr_call(S0, K, r, sigma, T, N):
    if N < 1:
        return black_scholes_call(S0, K, r, sigma, T)
    if T <= 0:
        return max(S0 - K, 0.0)

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
        payoff = disc * (p * payoff[1:] + (1.0 - p) * payoff[:-1])

    return float(payoff[0])

def mc_call_price(S0, K, r, sigma, T, M, seed=42, antithetic=True):
    if M < 1:
        return 0.0, 0.0, (0.0, 0.0)

    if T <= 0:
        price = max(S0 - K, 0.0)
        return price, 0.0, (price, price)

    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)

    if antithetic:
        half = (M + 1) // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])[:M]
    else:
        Z = rng.standard_normal(M)

    ST = S0 * np.exp(drift + vol * Z)
    payoff = np.maximum(ST - K, 0.0)
    disc_payoff = math.exp(-r * T) * payoff

    price = float(np.mean(disc_payoff))
    stderr = float(np.std(disc_payoff, ddof=1) / math.sqrt(M)) if M > 1 else 0.0
    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)
    return price, stderr, ci95

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")
st.title("European Call Option Pricing Dashboard")
st.caption("Black–Scholes vs Binomial Tree vs Monte Carlo")

with st.sidebar:
    st.header("Inputs")
    S0 = st.number_input("Spot S0", min_value=0.01, value=100.0)
    K = st.number_input("Strike K", min_value=0.01, value=100.0)
    r = st.number_input("Risk-free rate r", value=0.05)
    sigma = st.number_input("Volatility σ", min_value=0.0, value=0.20)
    T = st.number_input("Maturity T (years)", min_value=0.0, value=1.0)

    st.header("Binomial")
    N_tree = st.slider("Steps N", 1, 2000, 200)

    st.header("Monte Carlo")
    M = st.slider("Paths M", 100, 500000, 50000)
    antithetic = st.toggle("Antithetic variates", value=True)

bs = black_scholes_call(S0, K, r, sigma, T)
bt = binomial_crr_call(S0, K, r, sigma, T, N_tree)
mc, mc_se, mc_ci = mc_call_price(S0, K, r, sigma, T, M, antithetic=antithetic)

st.write("Black–Scholes Price:", bs)
st.write("Binomial Price:", bt)
st.write("Monte Carlo Price:", mc)
st.write("MC 95% CI:", mc_ci)