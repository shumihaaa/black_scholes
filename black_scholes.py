"""
black_scholes.py
Author: shumihaaa
Date: 2025-15-10
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Black–Scholes", layout="wide")
st.title("Black–Scholes Option Pricing Model")


# Black-Scholes functions

def _d1(S, K, T, r, sigma, q=0.0):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return np.nan
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma, q=0.0):
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_call(S, K, T, r, sigma, q=0.0):
    D1 = _d1(S, K, T, r, sigma, q)
    D2 = D1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def bs_put(S, K, T, r, sigma, q=0.0):
    D1 = _d1(S, K, T, r, sigma, q)
    D2 = D1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * np.exp(-q * T) * norm.cdf(-D1)


def bs_greeks(S, K, T, r, sigma, q=0.0):
    D1 = _d1(S, K, T, r, sigma, q)
    D2 = D1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(D1)
    delta_call = np.exp(-q * T) * norm.cdf(D1)
    delta_put = delta_call - np.exp(-q * T)
    gamma = (np.exp(-q * T) * pdf_d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
    theta_call = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(D2) + q * S * np.exp(-q * T) * norm.cdf(D1))
    theta_put = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-D2) - q * S * np.exp(-q * T) * norm.cdf(-D1))
    rho_call = K * T * np.exp(-r * T) * norm.cdf(D2)
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-D2)
    return {
        "d1": D1, "d2": D2,
        "delta_call": delta_call, "delta_put": delta_put,
        "gamma": gamma, "vega": vega,
        "theta_call": theta_call, "theta_put": theta_put,
        "rho_call": rho_call, "rho_put": rho_put
    }


# Implied volatility solver
def implied_vol(option_type, market_price, S, K, T, r, q=0.0, tol=1e-8, maxiter=200):
    """
    Solve for implied volatility using Brent's method.
    Returns np.nan if no solution found or invalid inputs.
    """
    try:
        market_price = float(market_price)
        if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
            return np.nan
    except Exception:
        return np.nan

    def price_diff(sigma):
        sigma = float(sigma)
        if sigma <= 0:
            sigma = 1e-12
        price = bs_call(S, K, T, r, sigma, q) if option_type == "call" else bs_put(S, K, T, r, sigma, q)
        return price - market_price

    low = 1e-8
    high = 5.0
    try:
        f_low = price_diff(low)
        f_high = price_diff(high)

        if f_low * f_high > 0:
            for ub in [10, 20, 50, 100]:
                f_high = price_diff(ub)
                if f_low * f_high <= 0:
                    high = ub
                    break

        if f_low * f_high > 0:
            return np.nan
        iv = brentq(price_diff, low, high, xtol=tol, maxiter=maxiter)
        return float(iv)
    except Exception:
        return np.nan


# Market data helpers (yfinance)
@st.cache_data(show_spinner=False)
def fetch_spot_and_history(ticker, hist_days=365):
    try:
        tk = yf.Ticker(ticker)
        end = date.today()
        start = end - timedelta(days=int(hist_days * 1.5))
        df = tk.history(start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            return None
        price_series = df['Close']
        spot = float(price_series.iloc[-1])
        return {"spot": spot, "history": price_series}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_option_chains(ticker):
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        chains = {}
        for e in expiries:
            oc = tk.option_chain(e)
            calls = oc.calls.copy()
            puts = oc.puts.copy()

            for df in (calls, puts):
                for col in ["strike", "bid", "ask", "lastPrice", "impliedVol"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            chains[e] = {"calls": calls, "puts": puts}
        return chains
    except Exception:
        return {}


def hist_vol(series, window_days=252):
    logret = np.log(series).diff().dropna()
    if len(logret) < 2:
        return np.nan
    vol_daily = logret.tail(window_days).std(ddof=1)
    return float(vol_daily * np.sqrt(252))


# UI: Sidebar inputs
st.sidebar.header("Mode & inputs")
mode = st.sidebar.selectbox("Mode", ["Manual", "Ticker (yfinance)"])

r = st.sidebar.number_input("Risk-free rate r (annual %)", value=2.0, step=0.1) / 100.0
q = st.sidebar.number_input("Dividend yield q (annual %, continuous)", value=0.0, step=0.1) / 100.0
days_to_expiry = st.sidebar.number_input("Days to expiry", min_value=1, value=90)
T = float(days_to_expiry) / 365.0

if mode == "Manual":
    S = st.sidebar.number_input("Spot S", value=100.0, format="%.4f")
    K = st.sidebar.number_input("Strike K", value=100.0, format="%.4f")
    sigma = st.sidebar.number_input("Volatility σ (annual %, set 0 to error)", value=20.0, step=0.1) / 100.0
else:
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL)").upper()
    hist_window = st.sidebar.number_input("History window for vol (days)", min_value=30, value=252)
    if not ticker:
        st.sidebar.info("Enter ticker to fetch market data")
        st.stop()
    data = fetch_spot_and_history(ticker, hist_days=hist_window)
    if data is None:
        st.sidebar.error("Failed to fetch ticker price/history")
        st.stop()
    S = data["spot"]
    st.sidebar.write(f"Spot (last close): {S:.4f}")
    K = st.sidebar.number_input("Strike K", value=float(round(S, 2)))
    sigma_input = st.sidebar.number_input("Volatility σ (annual %, 0 = auto historical)", value=0.0, step=0.1)
    if sigma_input == 0.0:
        est = hist_vol(data["history"], window_days=hist_window)
        sigma = est if not np.isnan(est) else 0.0
        st.sidebar.write(f"Estimated historical vol (annualized): {sigma:.2%}")
    else:
        sigma = sigma_input / 100.0

option_type = st.sidebar.radio("Option type", ("Call", "Put"))

st.sidebar.markdown("---")
st.sidebar.header("Implied Volatility")
use_iv_solver = st.sidebar.checkbox("Enable IV solver for single price", value=True)
market_option_price = st.sidebar.number_input("Market option price (for IV calc) — leave 0 if not used", value=0.0,
                                              min_value=0.0, format="%.6f")
compute_surface = st.sidebar.checkbox("Build implied vol surface from option chain (Ticker mode only)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Surface / visuals")
grid_res = st.sidebar.slider("Surface grid resolution", min_value=20, max_value=120, value=50)

# Input validation
if sigma <= 0 or S <= 0 or K <= 0 or T <= 0:
    st.error("Invalid inputs. Make sure S, K, σ, T are positive (σ can be auto-estimated in Ticker mode).")
    st.stop()

# Main display: inputs summary
st.subheader("Inputs summary")
st.markdown(f"- Spot (S): **{S:.4f}**")
st.markdown(f"- Strike (K): **{K:.4f}**")
st.markdown(f"- Time to expiry (T): **{T:.6f}** years ({days_to_expiry} days)")
st.markdown(f"- Volatility (σ): **{sigma:.2%}** (historical if auto-estimated)")
st.markdown(f"- Risk-free rate (r): **{r:.2%}**, Dividend yield (q): **{q:.2%}**")
st.markdown(f"- Option: **{option_type}**")

# Price, Greeks, implied vol single
st.markdown("## Black–Scholes Price & Greeks")
if option_type == "Call":
    model_price = bs_call(S, K, T, r, sigma, q)
else:
    model_price = bs_put(S, K, T, r, sigma, q)
gr = bs_greeks(S, K, T, r, sigma, q)

col1, col2, col3 = st.columns(3)
col1.metric("Model Price", f"${model_price:.6f}")
col2.metric("Delta (call/put)", f"{gr['delta_call']:.6f} / {gr['delta_put']:.6f}")
col3.metric("Gamma", f"{gr['gamma']:.6f}")
col4, col5, col6 = st.columns(3)
col4.metric("Vega", f"{gr['vega']:.6f}")
col5.metric("Theta (call/put)", f"{gr['theta_call']:.6f} / {gr['theta_put']:.6f}")
col6.metric("Rho (call/put)", f"{gr['rho_call']:.6f} / {gr['rho_put']:.6f}")

with st.expander("Detailed Greeks & internals"):
    dfg = pd.DataFrame({
        "metric": ["d1", "d2", "delta_call", "delta_put", "gamma", "vega", "theta_call", "theta_put", "rho_call",
                   "rho_put"],
        "value": [gr[k] for k in
                  ["d1", "d2", "delta_call", "delta_put", "gamma", "vega", "theta_call", "theta_put", "rho_call",
                   "rho_put"]],
        "description": ["d1", "d2", "Delta (call)", "Delta (put)", "Gamma", "Vega", "Theta call (yr)", "Theta put (yr)",
                        "Rho call", "Rho put"]
    })
    dfg["value"] = dfg["value"].map(lambda x: f"{x:.6f}")
    st.dataframe(dfg, height=300)

# Implied vol from a single market price
if use_iv_solver and market_option_price > 0:
    st.markdown("### Implied volatility (single market price)")
    opt = "call" if option_type == "Call" else "put"
    iv = implied_vol(opt, market_option_price, S, K, T, r, q)
    if np.isnan(iv):
        st.warning(
            "Implied vol could not be found for the provided market price (out of bounds or price inconsistent).")
    else:
        st.success(f"Implied volatility = {iv:.2%}  (market price = ${market_option_price:.6f})")

        model_at_iv = bs_call(S, K, T, r, iv, q) if opt == "call" else bs_put(S, K, T, r, iv, q)
        st.write(f"Model price at IV: ${model_at_iv:.6f} (should match market price)")

# Implied vol surface from option chain
if compute_surface and mode == "Ticker":
    st.markdown("## Implied Vol Surface from Option Chain")
    chains = fetch_option_chains(ticker)
    if not chains:
        st.warning("Option chain data not available via yfinance for this ticker.")
    else:
        rows = []
        for expiry, data in chains.items():
            days = (pd.to_datetime(expiry) - pd.to_datetime(date.today())).days
            if days <= 0:
                continue
            T_e = max(days / 365.0, 1 / 365.0)
            for kind in ("calls", "puts"):
                dfop = data[kind]
                if dfop is None or dfop.empty:
                    continue

                for _, r0 in dfop.iterrows():
                    strike = float(r0["strike"])
                    bid = r0.get("bid", np.nan)
                    ask = r0.get("ask", np.nan)
                    last = r0.get("lastPrice", np.nan)
                    mid = np.nan
                    if not np.isnan(bid) and not np.isnan(ask):
                        mid = 0.5 * (bid + ask)
                    elif not np.isnan(last):
                        mid = last
                    if mid is None or np.isnan(mid) or mid <= 0:
                        continue
                    iv = implied_vol("call" if kind == "calls" else "put", mid, S, strike, T_e, r, q)
                    rows.append(
                        {"expiry": expiry, "days": days, "T": T_e, "type": kind[:-1], "strike": strike, "mid": mid,
                         "impliedVol": iv})
        vol_df = pd.DataFrame(rows)
        vol_df = vol_df.dropna(subset=["impliedVol"])
        if vol_df.empty:
            st.warning("Could not compute implied vol for option chain entries (prices may be zero or inconsistent).")
        else:
            st.write(f"Computed implied vol for {len(vol_df)} option quotes.")
            st.dataframe(vol_df.head(250))

            calls_df = vol_df[vol_df["type"] == "call"].copy()
            if calls_df.empty:
                calls_df = vol_df.copy()
            strikes = np.sort(calls_df["strike"].unique())
            Ts = np.sort(calls_df["T"].unique())
            if strikes.size > 1 and Ts.size > 1:
                grid = np.full((Ts.size, strikes.size), np.nan)
                for i, tv in enumerate(Ts):
                    subset = calls_df[np.isclose(calls_df["T"], tv)]
                    for j, kv in enumerate(strikes):
                        rsub = subset[np.isclose(subset["strike"], kv)]
                        if not rsub.empty:
                            grid[i, j] = float(rsub["impliedVol"].mean())
                fig = go.Figure(data=[go.Surface(z=grid, x=strikes, y=Ts, colorscale="RdYlBu")])
                fig.update_layout(title=f"Implied Vol Surface for {ticker}",
                                  scene=dict(xaxis_title="Strike", yaxis_title="T (yrs)", zaxis_title="Implied Vol"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough distinct strikes or expiries to plot a full surface.")

# 3D Price surface
st.markdown("## Theoretical Price Surface (Strike vs Spot)")
S_min = max(0.01, S * 0.6)
S_max = S * 1.4
K_min = max(0.01, K * 0.6)
K_max = K * 1.4
n = int(grid_res)
S_vals = np.linspace(S_min, S_max, n)
K_vals = np.linspace(K_min, K_max, n)
S_g, K_g = np.meshgrid(S_vals, K_vals)
vec_price = np.vectorize(
    lambda s, k: bs_call(s, k, T, r, sigma, q) if option_type == "Call" else bs_put(s, k, T, r, sigma, q))
price_grid = vec_price(S_g, K_g)

fig = go.Figure(data=[go.Surface(x=S_g, y=K_g, z=price_grid, colorscale="Viridis")])
fig.update_layout(scene=dict(xaxis_title="Spot (S)", yaxis_title="Strike (K)", zaxis_title="Option Price"),
                  title="Theoretical Option Price Surface")
st.plotly_chart(fig, use_container_width=True)

# Greeks vs Spot
st.markdown("## Greeks vs Spot")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Price vs Strike (Spot fixed)")
    strikes = np.linspace(max(0.01, K * 0.5), K * 1.5, 200)
    prices_vs_K = [bs_call(S, k, T, r, sigma, q) if option_type == "Call" else bs_put(S, k, T, r, sigma, q) for k in
                   strikes]
    figk = go.Figure()
    figk.add_trace(go.Scatter(x=strikes, y=prices_vs_K, mode="lines"))
    figk.update_layout(xaxis_title="Strike", yaxis_title="Price", title=f"Price vs Strike (Spot={S:.2f})")
    st.plotly_chart(figk, use_container_width=True)

with col2:
    st.markdown("Greeks vs Spot (Strike fixed)")
    spots = np.linspace(max(0.01, S * 0.5), S * 1.5, 200)
    delta_vals = [
        bs_greeks(s, K, T, r, sigma, q)["delta_call"] if option_type == "Call" else bs_greeks(s, K, T, r, sigma, q)[
            "delta_put"] for s in spots]
    gamma_vals = [bs_greeks(s, K, T, r, sigma, q)["gamma"] for s in spots]
    vega_vals = [bs_greeks(s, K, T, r, sigma, q)["vega"] for s in spots]
    figg = go.Figure()
    figg.add_trace(go.Scatter(x=spots, y=delta_vals, name="Delta"))
    figg.add_trace(go.Scatter(x=spots, y=gamma_vals, name="Gamma"))
    figg.add_trace(go.Scatter(x=spots, y=vega_vals, name="Vega"))
    figg.update_layout(xaxis_title="Spot (S)", yaxis_title="Value", title=f"Greeks vs Spot (Strike={K:.2f})")
    st.plotly_chart(figg, use_container_width=True)

# Export surface
if st.button("Download theoretical price surface as CSV"):
    surf_df = pd.DataFrame(price_grid, index=[f"K={v:.4f}" for v in K_vals], columns=[f"S={v:.4f}" for v in S_vals])
    csv = surf_df.to_csv().encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"price_surface.csv", mime="text/csv")

st.caption(
    "This app computes Black–Scholes theoretical prices, Greeks and implied volatility. Implied vol solver uses Brent method and is applied to single-market prices or option chains (yfinance).")

