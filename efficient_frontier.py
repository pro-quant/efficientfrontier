import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

# Reference to Proquant
st.markdown(
    """
    
    To get started, you can download historical stock data using the,
    [Proquant Stock Data Downloader](https://proquant.se/apps/DownloadData/).
    """
)


st.markdown(
    "Upload a file (CSV or Excel) containing historical stock data to simulate portfolios, "
    "and calculate the Minimum Variance Portfolio (MVP) and Maximum Sharpe Ratio portfolio."
)

# File Upload
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file (columns as tickers, rows as dates, and adjusted closing prices).",
    type=["csv", "xlsx"],
)

# Risk-Free Rate Input
rf_rate = st.sidebar.number_input(
    "Risk-Free Rate (as a decimal, e.g., 0.02)", value=0.02, step=0.01)

if uploaded_file:
    try:
        # Determine file type and read data
        if uploaded_file.name.endswith(".csv"):
            stock_data = pd.read_csv(
                uploaded_file, index_col=0, parse_dates=True)
        elif uploaded_file.name.endswith(".xlsx"):
            stock_data = pd.read_excel(
                uploaded_file, index_col=0, parse_dates=True)

        st.write("### Uploaded Data Preview")
        st.dataframe(stock_data.head())

        # Process data
        returns = stock_data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Simulate portfolios
        with st.spinner("Simulating portfolios..."):
            weights, prets, pvols, pSharpe = simulate_portfolios(
                returns, mean_returns, cov_matrix, rf_rate)

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

            # Display Results in a Neat and Organized Manner
            st.subheader("Portfolio Optimization Results")

            # Maximum Sharpe Ratio Portfolio
            with st.expander("📈 Maximum Sharpe Ratio Portfolio"):
                st.markdown("**Portfolio Details:**")
                st.table(pd.DataFrame(
                    {
                        "Metric": ["Weights", "Return", "Volatility", "Sharpe Ratio"],
                        "Value": [
                            dict(zip(stock_data.columns, np.round(
                                max_sharpe_weights, 4))),
                            f"{max_sharpe_return:.4f}",
                            f"{max_sharpe_vol:.4f}",
                            f"{pSharpe[ind_max_sharpe]:.4f}"
                        ]
                    }
                ))
                st.markdown(f"""
                **Key Highlights:**
                - **Return**: {max_sharpe_return:.4%}
                - **Volatility (Risk)**: {max_sharpe_vol:.4%}
                - **Sharpe Ratio**: {pSharpe[ind_max_sharpe]:.2f}
                """)

            # Minimum Variance Portfolio
            with st.expander("📉 Minimum Variance Portfolio"):
                st.markdown("**Portfolio Details:**")
                st.table(pd.DataFrame(
                    {
                        "Metric": ["Weights", "Return", "Volatility", "Sharpe Ratio"],
                        "Value": [
                            dict(zip(stock_data.columns, np.round(mvp_weights, 4))),
                            f"{mvp_return:.4f}",
                            f"{mvp_vol:.4f}",
                            f"{pSharpe[ind_mvp]:.4f}"
                        ]
                    }
                ))
                st.markdown(f"""
                **Key Highlights:**
                - **Return**: {mvp_return:.4%}
                - **Volatility (Risk)**: {mvp_vol:.4%}
                - **Sharpe Ratio**: {pSharpe[ind_mvp]:.2f}
                """)

        # Plot Efficient Frontier
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(pvols, prets, c=pSharpe,
                             cmap="viridis", marker="o")
        ax.scatter(max_sharpe_vol, max_sharpe_return, color="red",
                   marker="*", s=200, label="Max Sharpe Ratio")
        ax.scatter(mvp_vol, mvp_return, color="blue",
                   marker="*", s=200, label="Minimum Variance")
        ax.set_xlabel("Portfolio Volatility (Risk)")
        ax.set_ylabel("Expected Portfolio Return")
        ax.set_title("Efficient Frontier: Expected Return vs. Volatility")
        ax.legend()
        fig.colorbar(scatter, label="Sharpe Ratio")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a valid file to proceed.")
