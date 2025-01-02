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
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=300,
                temperature=0.7
            )

            # Extract the text response from OpenAI
            factors_text = response.choices[0].message.content.strip()
            factors_list = []
            for factor in factors_text.split("\n"):
                if factor:
                    if ":" in factor:
                        factor_name, description = factor.split(":", 1)
                        factors_list.append({"Factor": factor_name.strip(), "Description": description.strip()})
                    else:
                        factors_list.append({"Factor": factor.strip(), "Description": "No description available."})

            # Store factors in a pandas DataFrame
            factors_df = pd.DataFrame(factors_list)
            st.write(factors_df)

            # Display the factors in a clear and concise manner
            st.header("Step 1: Factors Influencing Stock Price")
            st.markdown("Below are the key factors influencing the stock price of the company, listed in order of importance:")
            for idx, factor in enumerate(factors_list, start=1):
                st.markdown(f"{idx}. {factor['Factor']}: {factor['Description']}")

            # Display the original prompt and response
            st.header("Original ChatGPT Prompt and Response")
            st.subheader("Prompt")
            st.code(prompt, language="text")
            st.subheader("Response")
            st.code(factors_text, language="text")

            # Return the factors DataFrame for future use
            return factors_df

        except Exception as e:
            st.error(f"Error retrieving factors from OpenAI API: {e}")
            return pd.DataFrame()

# Step 5: Assign Ratings, Confidence Levels, and Importance Automatically
def assign_ratings_and_importance(factors_df):
    """
    Assigns ratings, confidence levels, and importance values to factors.

    Future Enhancement:
    - Incorporate Natural Language Processing (NLP) techniques to dynamically analyze descriptions for confidence scoring.
    - NLP could identify contextual nuances or conflicting signals in descriptions to refine confidence calculations.
    - Libraries like TextBlob, spaCy, or OpenAI GPT can assist with more robust implementations.
    """
    st.header("Step 2: Automatically Assign Ratings, Confidence Levels, and Importance")
    factor_list = factors_df['Factor'].tolist()
    descriptions = factors_df['Description'].tolist()

    ratings = []
    confidence_levels = []

    for description in descriptions:
        # Sentiment-based rating logic
        if any(word in description.lower() for word in ["increase", "growth", "profit"]):
            rating = "Positive"
        elif any(word in description.lower() for word in ["slightly positive", "moderate growth"]):
            rating = "Slightly Positive"
        elif any(word in description.lower() for word in ["slightly negative", "moderate decline"]):
            rating = "Slightly Negative"
        elif any(word in description.lower() for word in ["decline", "loss", "decrease"]):
            rating = "Negative"
        else:
            rating = "Neutral"

        # Calculate confidence based on consistency of descriptions
        sentiment_scores = []
        if "increase" in description or "growth" in description:
            sentiment_scores.append(1)
        elif "slightly positive" in description:
            sentiment_scores.append(0.5)
        elif "slightly negative" in description:
            sentiment_scores.append(-0.5)
        elif "decline" in description or "loss" in description:
            sentiment_scores.append(-1)
        else:
            sentiment_scores.append(0)

        # Variance calculation for consistency
        variance = np.var(sentiment_scores)

        # Assign confidence based on variance
        if variance < 0.2:
            confidence = 90
        elif variance < 0.5:
            confidence = 75
        else:
            confidence = 50

        ratings.append(rating)
        confidence_levels.append(confidence)

    return pd.DataFrame({
        "Factor": factor_list,
        "Rating": ratings,
        "Confidence (%)": confidence_levels
    })

# Step 6: Display Factors Table and Visualization Placeholder
def display_factors_table(factors_table):
    """
    Displays the factors table and visualizes importance and confidence levels.
    """
    st.header("Step 3: Factors Table")
    st.write(factors_table)

    # Placeholder for visualization
    st.header("Step 4: Visualization of Factors")
    st.markdown("Visualize the importance and confidence levels of the factors.")
    # Visualization will be added later

# Step 7: Critique and Analysis Placeholder
def critique_analysis(factors_table):
    """
    Allows the user to critique and make adjustments to the factors.
    """
    st.header("Step 5: Critique the Analysis")
    st.markdown("Review the table and make adjustments if needed.")
    # Display an editable version of the factors table
    edited_table = st.dataframe(factors_table)
    return edited_table

# Step 8: Make a Prediction Placeholder
def make_prediction(stock_name, duration):
    """
    Allows the user to input a prediction for the annualized rate of return.
    """
    st.header("Step 6: Make a Prediction")
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
        if not factors_df.empty:
            factors_table = assign_ratings_and_importance(factors_df)
            display_factors_table(factors_table)
            edited_table = critique_analysis(factors_table)
            make_prediction(stock_name, duration)

if __name__ == "__main__":
    main()
