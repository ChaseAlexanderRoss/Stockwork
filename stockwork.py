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
                f"Identify a single key factor influencing the stock price of {stock_name}. Include any relevant information pertaining to the single factor. Please output the response starting with the stock factor name, and then a description of the factor"
            )

            # Use the new gpt-4o-mini model for the API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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
            st.write(factors_df.style.set_properties(**{'white-space': 'pre-wrap'}))

            # Display the factors in a clear and concise manner
            st.header("Step 1: Factors Influencing Stock Price")
            st.markdown("Below are the key factors influencing the stock price of the company, listed in order of importance:")
            for idx, factor in enumerate(factors_list, start=1):
                st.markdown(f"{idx}. {factor['Factor']}: {factor['Description']}")

            # Display the original prompt and response
            st.header("Original ChatGPT Prompt and Response")
            st.subheader("Prompt")
            st.text(prompt)
            st.subheader("Response")
            st.text(factors_text)

            # Return the factors DataFrame for future use
            return factors_df

        except Exception as e:
            st.error(f"Error retrieving factors from OpenAI API: {e}")
            return pd.DataFrame()

# Step 5: Assign Ratings, Confidence Levels, and Importance Automatically
def assign_ratings_and_importance(factors_df):
    """
    Assigns ratings, confidence levels, and importance values to factors by feeding the factor description back to ChatGPT.
    """
    st.header("Step 2: Automatically Assign Ratings, Confidence Levels, and Importance")
    factor_list = factors_df['Factor'].tolist()
    descriptions = factors_df['Description'].tolist()

    ratings = []
    confidence_levels = []
    filtered_factors = []

    for factor, description in zip(factor_list, descriptions):
        if not description or description == "No description available.":
            continue

        try:
            # Construct the prompt to get rating and confidence for the factor
            prompt = (
                f"Take the following factor description and assign a rating (Very Positive, Positive, Neutral, Negative, Very Negative) to the description based on how that description would predict a stock to trend. Also, assign a confidence level (0-100%) based on the description, deciding how likely it is that the description is accurate to real life and how likely the prediction is to be true. The confidence level should be independent of the factor description. For example, if the factor is only neutral to the growth of the stock, but it is almost certainly true, then the confidence rating should be close to 100%\n\n"
                f"Description: {description}\n\n"
                f"Provide the response in the format: Rating: <rating>, Confidence: <confidence>%"
            )

            # Use the new gpt-4o-mini model for the API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=100,
                temperature=0.7
            )

            # Display the prompt and response before extracting the text
            st.subheader("Prompt")
            st.text(prompt)
            st.subheader("Response")
            st.text(response.choices[0].message.content.strip())

            # Extract the text response from OpenAI
            response_text = response.choices[0].message.content.strip()
            rating, confidence = response_text.split(", ")
            rating = rating.split(": ")[1]
            confidence = int(confidence.split(": ")[1].replace("%", ""))

            if rating != "Neutral" or confidence != 50:
                ratings.append(rating)
                confidence_levels.append(confidence)
                filtered_factors.append(factor)

        except Exception as e:
            st.error(f"Error retrieving rating and confidence from OpenAI API: {e}")

    return pd.DataFrame({
        "Factor": filtered_factors,
        "Rating": ratings,
        "Confidence (%)": confidence_levels
    })

# Step 6: Display Factors Table and Visualization Placeholder
def display_factors_table(factors_table):
    """
    Displays the factors table and visualizes importance and confidence levels.
    """
    st.header("Step 3: Factors Table")
    st.write(factors_table.style.set_properties(**{'white-space': 'pre-wrap'}).set_table_styles(
        [{'selector': 'th, td', 'props': [('max-width', '300px')]}]
    ))

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
    edited_table = st.dataframe(factors_table.style.set_properties(**{'white-space': 'pre-wrap'}).set_table_styles(
        [{'selector': 'th, td', 'props': [('max-width', '300px')]}]
    ))
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
