import streamlit as st
import pandas as pd
import yfinance as yf  # For retrieving financial data
import numpy as np

# Step 1: Ensure Required Libraries are Installed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    st.error("Please install the required libraries: matplotlib, seaborn.")

# Step 2: App Title and Description
st.title("Stock Prediction Analysis")
st.markdown(
    """
    This app helps you identify the key factors for a given stock, assign ratings and confidence levels, and assess their importance using the 100 pennies method.
    Follow the steps below to make an educated prediction for a company's stock price.
    """
)

# Step 3: User Inputs
stock_name = st.text_input("Enter the stock ticker symbol (e.g., NVDA):", "")
duration = st.number_input("Enter the time duration (in years) for prediction:", min_value=1, max_value=10, value=1)

# Step 4: Automatically Generate Factors
if stock_name:
    st.header("Step 1: Identify Factors")
    st.markdown(f"Identifying the key factors for predicting the stock price of {stock_name} over {duration} year(s).")
    
    # Retrieve stock data using yfinance
    stock = yf.Ticker(stock_name)
    try:
        stock_info = stock.info
        sector = stock_info.get("sector", "Unknown Sector")
        market_cap = stock_info.get("marketCap", 0)
        pe_ratio = stock_info.get("trailingPE", np.nan)
        revenue_growth = stock_info.get("revenueGrowth", np.nan)
    except Exception as e:
        st.error("Unable to retrieve stock information. Please check the stock ticker symbol.")
        stock_info = None

    if stock_info:
        # Generate factors based on the duration and stock characteristics
        if duration <= 1:
            factor_list = [
                "Quarterly Earnings Reports",
                "Market Sentiment",
                "Competitor Performance",
                "Short-term Economic Indicators",
                "Recent Product Launches"
            ]
        else:
            factor_list = [
                "Long-term Industry Trends",
                "Product Innovation and R&D",
                "Market Expansion Strategies",
                "Economic Cycles",
                "Regulatory Environment",
                "Competitive Landscape",
                "Company Financial Health"
            ]

        # Customize factors based on sector
        if sector == "Technology":
            factor_list.append("Adoption of New Technologies")
        elif sector == "Healthcare":
            factor_list.append("Regulatory Approvals and R&D Breakthroughs")
        elif sector == "Financial Services":
            factor_list.append("Interest Rate Changes and Regulatory Policies")

        # Step 5: Assign Ratings, Confidence, and Importance Automatically
        st.header("Step 2: Automatically Assign Ratings, Confidence Levels, and Importance")
        ratings = []
        confidence_levels = []
        importance_values = []
        total_pennies = 100

        for factor in factor_list:
            # Assigning ratings based on sector-specific analysis
            if "Economic" in factor or "Regulatory" in factor:
                rating = "Neutral"
                confidence = 75 if duration > 1 else 60
            elif "Competitor" in factor or "Market Sentiment" in factor:
                rating = "Positive" if market_cap > 1e10 else "Neutral"
                confidence = 80 if market_cap > 1e10 else 65
            elif "Product Innovation" in factor or "R&D" in factor:
                rating = "Very Positive" if revenue_growth > 0.1 else "Positive"
                confidence = 90 if revenue_growth > 0.1 else 70
            elif "Adoption of New Technologies" in factor:
                rating = "Very Positive"
                confidence = 85
            elif "Regulatory Approvals" in factor:
                rating = "Neutral" if pe_ratio > 20 else "Positive"
                confidence = 70
            else:
                rating = "Positive"
                confidence = 75

            # Importance based on perceived impact of factor
            if "Company Financial Health" in factor or "Economic Cycles" in factor:
                importance = 20
            elif "Product Innovation" in factor or "R&D" in factor:
                importance = 15
            else:
                importance = total_pennies // len(factor_list)
            total_pennies -= importance

            ratings.append(rating)
            confidence_levels.append(confidence)
            importance_values.append(importance)

        # Step 6: Create Factors Table
        st.header("Step 3: Factors Table")
        data = {
            "Factor": factor_list,
            "Rating": ratings,
            "Confidence (%)": confidence_levels,
            "Importance (pennies)": importance_values
        }
        factors_table = pd.DataFrame(data)
        st.write(factors_table)
        
        # Step 7: Visualization of Factors
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.header("Step 4: Visualization of Factors")
            st.markdown("Visualize the importance and confidence levels of the factors.")
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            sns.barplot(x="Importance (pennies)", y="Factor", data=factors_table, ax=ax[0], palette="viridis")
            ax[0].set_title("Importance of Factors")
            ax[0].set_xlabel("Importance (pennies)")
            ax[0].set_ylabel("Factor")
            
            sns.barplot(x="Confidence (%)", y="Factor", data=factors_table, ax=ax[1], palette="plasma")
            ax[1].set_title("Confidence Levels of Factors")
            ax[1].set_xlabel("Confidence (%)")
            ax[1].set_ylabel("Factor")
            
            st.pyplot(fig)
        except ModuleNotFoundError:
            st.error("Visualization requires matplotlib and seaborn. Please install these libraries.")
        
        # Step 8: Critique and Analysis
        st.header("Step 5: Critique the Analysis")
        st.markdown("Review the table and make adjustments if needed.")
        # Display an editable version of the factors table
        edited_table = st.dataframe(factors_table)
        
        # Step 9: Make a Prediction
        st.header("Step 6: Make a Prediction")
        st.markdown("Based on the data in the table, make a prediction for the annualized rate of return.")
        predicted_return = st.number_input(
            "Enter your prediction for the annualized rate of return (%):",
            min_value=-100.0,
            max_value=100.0,
            value=0.0
        )
        
        st.markdown(
            f"**Predicted Annualized Rate of Return for {stock_name}: {predicted_return}% over {duration} year(s)**"
        )
