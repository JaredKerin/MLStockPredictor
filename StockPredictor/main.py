import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 
import pandas as pd

# Fetch historical data for the S&P 500 index
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

# Remove unnecessary columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create a new column for the next day's closing price
sp500["Tommorow"] = sp500["Close"].shift(-1)

# Create a binary target column where 1 means the price increased the next day and 0 otherwise
sp500["Target"] = (sp500["Tommorow"] > sp500["Close"]).astype(int)

# Focus on data from 1990 onwards
sp500 = sp500.loc["1990-01-01":].copy()

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Split the data into training and testing sets
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Define the predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train the model using the training data
model.fit(train[predictors], train["Target"])

# Make predictions on the testing data
preds = model.predict(test[predictors])

# Convert predictions to a Pandas Series
preds = pd.Series(preds, index=test.index)

# Calculate the precision score of the predictions
precision_score(test["Target"], preds)

# Combine the actual target values and predictions for comparison
combined = pd.concat([test["Target"], preds], axis=1)

# Function to predict market movement
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Function to backtest the model
def backtest(data, model, predictors, start=2500, step=2500):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Backtest the model
predictions = backtest(sp500, model, predictors)

# Define different time horizons for rolling averages
horizons = [2, 5, 250, 1000]

new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

# Drop rows with missing values
sp500 = sp500.dropna()

# Backtest the model with new predictors
predictions = backtest(sp500, model, new_predictors)

#Prints the amount on times the market will go up or down
#print(predictions["Predictions"].value_counts())

#Prints Precision score
#print(precision_score(predictions["Target"], predictions["Predictions"]))

#Prints percentage of days where the market actually went up
#print(predictions["Target"].value_counts() / predictions.shape[0])

#print(sp500)
#print(sp500.index)