
# Import the required libraries
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from pandas import Timestamp


def predict_future_price(crypto_type, model_fit, train, date):
    model_pred = []
    predictions = []

#     for date in future_dates:
    train_data = train.copy()  # Create a copy of the training data for each iteration

    # Make prediction for the future date
    forecast = model_fit.forecast(steps=1)
    prediction = forecast[0]
    train_data.append(prediction)  # Update the copy of the training data with the predicted value

    # Retrain the model with the updated training data
    model = sm.tsa.arima.ARIMA(train_data, order=(4, 1, 0))
    model_fit = model.fit()
    model_pred.append(prediction)
    predictions.append((date, prediction))

    # Create a DataFrame for side-by-side comparison
    df_predictions = pd.DataFrame(predictions, columns=['Date', 'Prediction'])
    df_predictions.set_index('Date', inplace=True)

    return model_pred, df_predictions, model_fit

# Function to simulate trading decisions
def simulate_trading(predictions, latest):
    # Define trading logic based on predictions
    trading_decision = None
    price_difference = predictions - latest
    if abs(price_difference) <= 25:
        trading_decision = "Hold"
    elif price_difference > 25:
        trading_decision = "Sell"
    else:
        trading_decision = "Buy"
    return trading_decision

# Main app code
# Main app code
def main(Date, Coin):
    with open("bmodel.pkl", "rb") as file:
        bmodel = pickle.load(file)
    with open("emodel.pkl", "rb") as file:
        emodel = pickle.load(file)
    with open("lmodel.pkl", "rb") as file:
        lmodel = pickle.load(file)
    df= pd.read_csv('crypto.csv', index_col=False)
    
    ##Bitcoin
    bdf = df[df['Coin'] == 'Bitcoin']
    bdf = bdf.set_index('Date')
    bdf= bdf['Close']
    b = int(len(bdf) * 0.9)
    btrain = list(bdf[0:b])
    btest = list(bdf[b:])

    ##ethereum
    edf = df[df['Coin'] == 'Ethereum']
    edf = edf.set_index('Date')
    edf= edf['Close']
    e = int(len(edf) * 0.9)
    etrain = list(edf[0:e])
    etest = list(edf[e:])

    ##Litecoin
    ldf = df[df['Coin'] == 'Litecoin']
    ldf = ldf.set_index('Date')
    ldf= ldf['Close']
    l = int(len(ldf) * 0.9)
    ltrain = list(ldf[0:l])
    ltest = list(ldf[l:])
  
    if Coin == 'Bitcoin':
        train = btrain
        model_fit = bmodel
        predictions, df_predictions, bmodel = predict_future_price(Coin, model_fit, train, Date)
        latest = bdf[-1]
        with open("bmodel.pkl", "wb") as file:
            pickle.dump(bmodel, file)
    elif Coin == 'Ethereum':
        train = etrain
        predictions, df_predictions, emodel = predict_future_price(Coin, emodel, train, Date)
        latest = edf[-1]
        with open("emodel.pkl", "wb") as file:
            pickle.dump(emodel, file)

    elif Coin == 'Litecoin':
        train = ltrain
        model_fit = lmodel
        predictions, df_predictions, lmodel = predict_future_price(Coin, model_fit, train, Date)
        latest = ldf[-1]
        with open("lmodel.pkl", "wb") as file:
            pickle.dump(lmodel, file)
    else:
        raise ValueError("Invalid cryptocurrency type. Choose one from 'Bitcoin', 'Ethereum', or 'Litecoin'.")


    trading_decisions = simulate_trading(predictions[0], latest )

    return predictions[0], trading_decisions

