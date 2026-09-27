"""
Black-Scholes Option Pricing — Data Pull + Pricing
Requires: pip install yfinance pandas_datareader numpy scipy
"""

import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas_datareader.data as web
import datetime

# ----------------------------
# 1. PULL DATA
# ----------------------------
print('a')
def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1d")["Close"].iloc[-1]

def get_historical_volatility(ticker, lookback_days=252):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1))
    return log_returns.std() * np.sqrt(252)

def get_risk_free_rate():
    start = datetime.datetime.now() - datetime.timedelta(days=30)
    r_data = web.DataReader("DGS3MO", "fred", start)
    return r_data.dropna().iloc[-1].values[0] / 100

def get_option_chain(ticker, expiration_index=0):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    exp_date = expirations[expiration_index]
    chain = stock.option_chain(exp_date)
    return chain, exp_date


# ----------------------------
# 2. BLACK-SCHOLES FORMULA
# ----------------------------

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S: current stock price
    K: strike price
    T: time to expiration (in years)
    r: risk-free rate (decimal, e.g. 0.05)
    sigma: volatility (decimal, e.g. 0.20)
    option_type: "call" or "put"
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


# ----------------------------
# 3. GREEKS (bonus)
# ----------------------------

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol change
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))
    ) / 365  # per day
    rho = (
        K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))
    ) / 100  # per 1% rate change

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ----------------------------
# 4. FULL PIPELINE EXAMPLE
# ----------------------------

if __name__ == "__main__":
    ticker = "AAPL"
    strike = 230.0
    days_to_expiry = 30

    S = get_stock_price(ticker)
    sigma = get_historical_volatility(ticker)
    r = get_risk_free_rate()
    T = days_to_expiry / 365

    call_price = black_scholes(S, strike, T, r, sigma, "call")
    put_price = black_scholes(S, strike, T, r, sigma, "put")
    greeks = black_scholes_greeks(S, strike, T, r, sigma, "call")

    print(f"Ticker: {ticker}")
    print(f"Stock Price (S): {S:.2f}")
    print(f"Strike (K): {strike}")
    print(f"Time to Expiry (T): {T:.4f} years")
    print(f"Risk-Free Rate (r): {r:.4%}")
    print(f"Volatility (sigma): {sigma:.4%}")
    print(f"\nCall Price: {call_price:.2f}")
    print(f"Put Price: {put_price:.2f}")
    print(f"Greeks (call): {greeks}")
