import streamlit as st
import pandas as pd
import yfinance as yf  # For retrieving financial data
import numpy as np
import os

# Step 1.1: Ensure OpenAI Library is Installed
try:
    from openai import OpenAI
except ModuleNotFoundError:
    st.error("Please install the required library: openai.")
    raise


# Set OpenAI API Key (replace with your actual API key)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Step 2: App Title and Description
st.title("Stock Prediction Analysis")
st.markdown(
    """
    This app helps you identify the key factors for a given stock, assign ratings and confidence levels, and assess their importance using the 100 pennies method.
    Follow the steps below to make an educated prediction for a company's stock price.
    """
)

# Step 3: User Inputs
def get_user_inputs():
    """
    Collects user inputs for stock ticker and prediction duration.
    """
    stock_name = st.text_input("Enter the stock ticker symbol (e.g., NVDA):", "")
    duration = st.selectbox(
        "Select the prediction duration:",
        ("1 Month", "3 Months", "6 Months", "1 Year"),
    )
    return stock_name, duration

# Step 4: Fetch Stock Data
def fetch_stock_data(ticker):
    """
    Retrieves historical stock data using yfinance.
    """
    try:
        stock_data = yf.download(ticker, period="1y")
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Step 5: Analysis and Visualization
def analyze_stock_data(stock_data):
    """
    Analyzes and visualizes stock data.
    """
    if stock_data is not None:
        st.line_chart(stock_data["Close"], use_container_width=True)
        st.write("Data Summary:")

        # Format and display the data
        formatted_data = stock_data.copy()
        formatted_data.index = formatted_data.index.strftime("%Y-%m-%d")  # Format dates
        formatted_data = formatted_data.round(2)  # Round numeric values
        st.dataframe(formatted_data)

        # Highlight statistics
        stats = stock_data.describe().round(2)
        st.write("Descriptive Statistics:")
        st.dataframe(stats.style.highlight_max(axis=0))
    else:
        st.warning("No data available for analysis.")

# Step 6: Main App Logic
def main():
    stock_name, duration = get_user_inputs()

    if stock_name:
        st.subheader(f"Stock Analysis for: {stock_name}")
        stock_data = fetch_stock_data(stock_name)
        analyze_stock_data(stock_data)
    else:
        st.info("Please enter a stock ticker to start analysis.")

if __name__ == "__main__":
    main()

