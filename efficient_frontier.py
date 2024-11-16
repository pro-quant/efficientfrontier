import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta


# Function to fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    adjusted_closes = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)[["Adj Close"]]
            if df.empty:
                st.error(f"No data found for {ticker}. Check the ticker symbol or adjust the date range.")
                return None
            # Ensure dates are timezone-naive
            df.index = df.index.tz_localize(None)
            adjusted_closes[ticker] = df["Adj Close"]
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
    return pd.DataFrame(adjusted_closes)


# Portfolio optimization
def simulate_portfolios(returns, mean_returns, cov_matrix, rf_rate, portfolio_amount, num_simulations):
    noa = len(returns.columns)
    pWeights, prets, pvols, pSharpe = [], [], [], []

    for _ in range(num_simulations):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        pWeights.append(weights)

        ret = np.sum(mean_returns * weights) * 252
        vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
        sharpe = (ret - rf_rate) / vol

        prets.append(ret * portfolio_amount / 100)
        pvols.append(vol * portfolio_amount / 100)
        pSharpe.append(sharpe)

    return np.array(pWeights), np.array(prets), np.array(pvols), np.array(pSharpe)


# Streamlit UI
st.title("Portfolio Optimization with Efficient Frontier")
st.markdown(
    "This app simulates portfolios and calculates the Minimum Variance Portfolio (MVP) and Maximum Sharpe Ratio Portfolio."
)

# Sidebar inputs
st.sidebar.header("Portfolio Parameters")
end_date = datetime.today()
start_date = end_date - timedelta(days=3*365)
tickers = st.sidebar.text_input("Enter 2 or 3 Ticker Symbols (comma-separated)", value="AAPL, MSFT, META")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

portfolio_amount = st.sidebar.number_input("Portfolio Amount (e.g., 1000)", value=1000.0, step=100.0)
rf_rate = st.sidebar.number_input("Risk-Free Rate (as a decimal, e.g., 0.02)", value=0.02, step=0.01)
num_simulations = st.sidebar.number_input("Number of Portfolios to Simulate", value=1000, min_value=100, step=100)

if len(tickers) < 2 or len(tickers) > 3:
    st.error("Please enter exactly 2 or 3 ticker symbols.")
else:
    if st.button("Simulate Portfolio"):
        with st.spinner("Fetching stock data and simulating portfolios..."):
            stock_data = fetch_stock_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if stock_data is not None:
                returns = stock_data.pct_change().dropna()
                mean_returns = returns.mean()
                cov_matrix = returns.cov()

                weights, prets, pvols, pSharpe = simulate_portfolios(
                    returns, mean_returns, cov_matrix, rf_rate, portfolio_amount, num_simulations
                )

                ind_max_sharpe = np.argmax(pSharpe)
                max_sharpe_weights = weights[ind_max_sharpe]
                max_sharpe_return = prets[ind_max_sharpe]
                max_sharpe_vol = pvols[ind_max_sharpe]

                ind_mvp = np.argmin(pvols)
                mvp_weights = weights[ind_mvp]
                mvp_return = prets[ind_mvp]
                mvp_vol = pvols[ind_mvp]

                st.subheader("Portfolio Optimization Results")
                st.write(f"**Portfolio Amount: ${portfolio_amount:.2f}**")
                st.write(f"**Maximum Sharpe Ratio Portfolio**")
                st.write(f"Weights: {dict(zip(tickers, np.round(max_sharpe_weights, 4)))}")
                st.write(f"Return: ${max_sharpe_return:.2f}")
                st.write(f"Volatility: ${max_sharpe_vol:.2f}")
                st.write(f"Sharpe Ratio: {pSharpe[ind_max_sharpe]:.4f}")

                st.write(f"**Minimum Variance Portfolio**")
                st.write(f"Weights: {dict(zip(tickers, np.round(mvp_weights, 4)))}")
                st.write(f"Return: ${mvp_return:.2f}")
                st.write(f"Volatility: ${mvp_vol:.2f}")
                st.write(f"Sharpe Ratio: {pSharpe[ind_mvp]:.4f}")

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
