# MLStockPredictor
Machine Learning algorithm to help predict changes in the SP500. 

Summary
This script analyzes historical stock data for the S&P 500 index to predict whether the closing price will increase the next day. Here's a breakdown of the main steps:

Data Retrieval: 
Uses the yfinance library to fetch historical data for the S&P 500 index.

Data Preparation:
Removes unnecessary columns (Dividends and Stock Splits).
Adds a column for the next day's closing price.
Creates a binary target column indicating whether the price increased the next day.
Filters the data to start from 1990.

Model Training and Testing:
Initializes a RandomForestClassifier.
Splits the data into training and testing sets.
Trains the model using the training data and predefined predictors (Close, Volume, Open, High, Low).
Makes predictions on the testing data and calculates the precision score.
Prediction Function:
Defines a function to train the model and make predictions with a threshold to classify the probability of an increase.

Backtesting:
Implements a backtesting function to evaluate the model's performance over different periods.

Enhanced Predictors:
Introduces new predictors based on rolling averages and trends over various time horizons.
Retrains and backtests the model with these enhanced predictors.

Evaluation:
(Commented out on original script) Prints the number of days the market prediction was up or down, precision score, and percentage of actual market increases.

Visualization:
Use web framework Flask to build webpage to present the data.
