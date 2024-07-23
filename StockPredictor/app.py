from flask import Flask, render_template
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch historical data for the S&P 500 index
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")

    # Data preparation
    del sp500["Dividends"]
    del sp500["Stock Splits"]
    sp500["Tommorow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tommorow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()

    # Model training and testing
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    precision = precision_score(test["Target"], preds)
    
    # Backtesting
    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:,1]
        preds[preds >= .6] = 1
        preds[preds < .6] = 0
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined

    def backtest(data, model, predictors, start=2500, step=2500):
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions = predict(train, test, predictors, model)
            all_predictions.append(predictions)
        return pd.concat(all_predictions)
    
    horizons = [2, 5, 250, 1000]
    new_predictors = []
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column, trend_column]

    sp500 = sp500.dropna()
    predictions = backtest(sp500, model, new_predictors)
    prediction_counts = predictions["Predictions"].value_counts().to_dict()
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    target_percentages = (predictions["Target"].value_counts() / predictions.shape[0]).to_dict()

    # Render results in the template
    return render_template('index.html', precision=precision, prediction_counts=prediction_counts, target_percentages=target_percentages)

if __name__ == '__main__':
    app.run(debug=True)
