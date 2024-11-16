import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf

# Fix for pandas_datareader with yfinance
yf.pdr_override()

# Function to fetch stock data
def fetch_stock_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            df = pdr.get_data_yahoo(ticker, start="2015-01-01", end="2023-12-31")
            data[ticker] = df["Adj Close"]
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
    return pd.DataFrame(data)

# Portfolio optimization
def simulate_portfolios(returns, mean_returns, cov_matrix, rf_rate):
    noa = len(returns.columns)  # Number of assets
    pWeights, prets, pvols, pSharpe = [], [], [], []

    # Generate random portfolios
    for _ in range(5000):
        weights = np.random.random(noa)
        weights /= np.sum(weights)  # Normalize weights to sum to 1
        pWeights.append(weights)

        # Portfolio return and volatility
        ret = np.sum(mean_returns * weights) * 252
        vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
        sharpe = (ret - rf_rate) / vol  # Sharpe ratio

        prets.append(ret)
        pvols.append(vol)
        pSharpe.append(sharpe)

    return np.array(pWeights), np.array(prets), np.array(pvols), np.array(pSharpe)

# Streamlit UI
st.title("Portfolio Optimization and Simulation")
st.markdown(
    "This app allows you to input 2 or 3 ticker symbols, simulates portfolios, and calculates the Minimum Variance Portfolio (MVP) and Maximum Sharpe Ratio portfolio."
)

# User Inputs
st.sidebar.header("Portfolio Parameters")
tickers = st.sidebar.text_input("Enter 2 or 3 Ticker Symbols (comma-separated)").split(",")
rf_rate = st.sidebar.number_input("Risk-Free Rate (as a decimal, e.g., 0.02)", value=0.02, step=0.01)

# Ensure valid input
if len(tickers) < 2 or len(tickers) > 3:
    st.error("Please enter exactly 2 or 3 ticker symbols.")
else:
    if st.button("Simulate Portfolio"):
        with st.spinner("Fetching stock data and simulating portfolios..."):
            stock_data = fetch_stock_data([ticker.strip().upper() for ticker in tickers])

            if stock_data is not None:
                returns = stock_data.pct_change().dropna()
                mean_returns = returns.mean()
                cov_matrix = returns.cov()

                # Perform portfolio simulation
                weights, prets, pvols, pSharpe = simulate_portfolios(returns, mean_returns, cov_matrix, rf_rate)

                # Find the portfolio with the highest Sharpe Ratio
                ind_max_sharpe = np.argmax(pSharpe)
                max_sharpe_weights = weights[ind_max_sharpe]
                max_sharpe_return = prets[ind_max_sharpe]
                max_sharpe_vol = pvols[ind_max_sharpe]

                # Find the Minimum Variance Portfolio
                ind_mvp = np.argmin(pvols)
                mvp_weights = weights[ind_mvp]
                mvp_return = prets[ind_mvp]
                mvp_vol = pvols[ind_mvp]

                # Display Results
                st.subheader("Portfolio Optimization Results")
                st.write(f"**Maximum Sharpe Ratio Portfolio**")
                st.write(f"Weights: {dict(zip(tickers, np.round(max_sharpe_weights, 4)))}")
                st.write(f"Return: {max_sharpe_return:.4f}")
                st.write(f"Volatility: {max_sharpe_vol:.4f}")
                st.write(f"Sharpe Ratio: {pSharpe[ind_max_sharpe]:.4f}")

                st.write(f"**Minimum Variance Portfolio**")
                st.write(f"Weights: {dict(zip(tickers, np.round(mvp_weights, 4)))}")
                st.write(f"Return: {mvp_return:.4f}")
                st.write(f"Volatility: {mvp_vol:.4f}")
                st.write(f"Sharpe Ratio: {pSharpe[ind_mvp]:.4f}")

                # Efficient Frontier Plot
                if st.button("Plot Efficient Frontier"):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    scatter = ax.scatter(pvols, prets, c=pSharpe, cmap="viridis", marker="o")
                    ax.scatter(max_sharpe_vol, max_sharpe_return, color="red", marker="*", s=200, label="Max Sharpe Ratio")
                    ax.scatter(mvp_vol, mvp_return, color="blue", marker="*", s=200, label="Minimum Variance")
                    ax.set_xlabel("Portfolio Volatility (Risk)")
                    ax.set_ylabel("Expected Portfolio Return")
                    ax.set_title("Efficient Frontier: Expected Return vs. Volatility")
                    ax.legend()
                    fig.colorbar(scatter, label="Sharpe Ratio")
                    st.pyplot(fig)
