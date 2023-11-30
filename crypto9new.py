
# Import the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import crypto8



# Main app code
def main():
    st.title("Crypto Price Prediction and Trading Bot")
    st.write("Enter the required inputs and generate predictions.")
    # User inputs
    Date = st.date_input("Date")
    crypto_options = {
    "Bitcoin",
    "Ethereum",
    "Litecoin"
    }

    # Create a dropdown for selecting the cryptocurrency
    Coin = st.selectbox("Coin", options=list(crypto_options))
    
    # Call the main function
    results = crypto8.main(Date, Coin)

    # Retrieve the predictions and trading decisions
    predictions = results[0]
    trading_decisions = results[1]
    
    #Outputs
    st.header("Future Price Predictions")
    st.write("Price:", predictions)

    st.header("Trading Decisions")
    st.write("Decision:", trading_decisions)


# Run the app
if __name__ == "__main__":
    main()
