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
    duration = st.number_input("Enter the time duration (in years) for prediction:", min_value=1, max_value=10, value=1)
    return stock_name, duration

# Step 4: Retrieve Important Factors using OpenAI API
def generate_stock_summary(stock_name):
    """
    Uses the OpenAI API to generate a list of significant factors affecting the stock price of the given company.
    The summary includes recent financial performance, market trends, significant news, etc.
    """
    if stock_name:
        st.header("Stock Summary Analysis")
        try:
            # Construct the prompt to retrieve key factors influencing the stock price
            prompt = (
                f"Identify the key factors influencing the stock price of {stock_name}. Include aspects such as financial performance, market trends, industry news, economic indicators, regulatory changes, and any other relevant information."
            )

            # Use the new gpt-4o model for the API call
            response = client.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )

            # Extract the text response from OpenAI
            factors_text = response.choices[0].message['content'].strip()
            factors_list = []
            for factor in factors_text.split("\n"):
                if factor:
                    if ":" in factor:
                        factor_name, description = factor.split(":", 1)
                        factors_list.append({"Factor": factor_name.strip(), "Description": description.strip()})
                    else:
                        factors_list.append({"Factor": factor.strip(), "Description": "No description available."})

            # Order factors by importance (placeholder logic - actual ranking can be added based on additional analysis)
            factors_list = sorted(factors_list, key=lambda x: len(x['Description']), reverse=True)

            # Store factors in a pandas DataFrame
            factors_df = pd.DataFrame(factors_list)
            st.write(factors_df)

            # Display the factors in a clear and concise manner
            st.header("Step 1: Factors Influencing Stock Price")
            st.markdown("Below are the key factors influencing the stock price of the company, listed in order of importance:")
            for idx, factor in enumerate(factors_list, start=1):
                st.markdown(f"**{idx}. {factor['Factor']}**: {factor['Description']}")

            # Return the factors DataFrame for future use
            return factors_df

        except Exception as e:
            st.error(f"Error retrieving factors from OpenAI API: {e}")
            return pd.DataFrame()

# Step 5: Automatically Generate Factors
def generate_factors(stock_name, duration):
    """
    Generates factors for predicting the stock price based on user inputs.
    """
    st.header("Step 2: Identify Additional Factors")
    st.markdown(f"Identifying additional key factors for predicting the stock price of {stock_name} over {duration} year(s).")

    # Retrieve stock data using yfinance
    stock = yf.Ticker(stock_name)
    try:
        stock_info = stock.info
        sector = stock_info.get("sector", "Unknown Sector")
    except Exception as e:
        st.error("Unable to retrieve stock information. Please check the stock ticker symbol.")
        return []

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

    return factor_list

# Step 6: Assign Ratings, Confidence Levels, and Importance Automatically
def assign_ratings_and_importance(factor_list, duration):
    """
    Assigns ratings, confidence levels, and importance values to factors.
    """
    st.header("Step 3: Automatically Assign Ratings, Confidence Levels, and Importance")
    ratings = []
    confidence_levels = []
    importance_values = []
    total_pennies = 100

    for factor in factor_list:
        # Placeholder logic for assigning ratings, confidence, and importance
        rating = "Neutral"
        confidence = 75
        importance = total_pennies // len(factor_list)
        total_pennies -= importance

        ratings.append(rating)
        confidence_levels.append(confidence)
        importance_values.append(importance)

    return pd.DataFrame({
        "Factor": factor_list,
        "Rating": ratings,
        "Confidence (%)": confidence_levels,
        "Importance (pennies)": importance_values
    })

# Step 7: Display Factors Table and Visualization Placeholder
def display_factors_table(factors_table):
    """
    Displays the factors table and visualizes importance and confidence levels.
    """
    st.header("Step 4: Factors Table")
    st.write(factors_table)

    # Placeholder for visualization
    st.header("Step 5: Visualization of Factors")
    st.markdown("Visualize the importance and confidence levels of the factors.")
    # Visualization will be added later

# Step 8: Critique and Analysis Placeholder
def critique_analysis(factors_table):
    """
    Allows the user to critique and make adjustments to the factors.
    """
    st.header("Step 6: Critique the Analysis")
    st.markdown("Review the table and make adjustments if needed.")
    # Display an editable version of the factors table
    edited_table = st.dataframe(factors_table)
    return edited_table

# Step 9: Make a Prediction Placeholder
def make_prediction(stock_name, duration):
    """
    Allows the user to input a prediction for the annualized rate of return.
    """
    st.header("Step 7: Make a Prediction")
    st.markdown("Based on the data in the table, make a prediction for the annualized rate of return.")
    predicted_return = st.number_input(
        "Enter your prediction for the annualized rate of return (%):",
        min_value=-100.0,
        max_value=100.0,
        value=0.0
    )
    st.markdown(f"**Predicted Annualized Rate of Return for {stock_name}: {predicted_return}% over {duration} year(s)**")

# Main App Flow
def main():
    """
    Main function to run the stock predictor app.
    """
    stock_name, duration = get_user_inputs()
    if stock_name:
        factors_df = generate_stock_summary(stock_name)
        additional_factors = generate_factors(stock_name, duration)
        if additional_factors:
            factors_df = pd.concat([factors_df, pd.DataFrame({"Factor": additional_factors})], ignore_index=True)
            factors_table = assign_ratings_and_importance(factors_df["Factor"].tolist(), duration)
            display_factors_table(factors_table)
            edited_table = critique_analysis(factors_table)
            make_prediction(stock_name, duration)

if __name__ == "__main__":
    main()

