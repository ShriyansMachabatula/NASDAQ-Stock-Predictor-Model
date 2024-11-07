import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load or fetch NASDAQ Composite historical data
if os.path.exists("nasdaq.csv"):
    nasdaq = pd.read_csv("nasdaq.csv", index_col=0)
else:
    nasdaq = yf.Ticker("^IXIC")  # NASDAQ Composite ticker
    nasdaq = nasdaq.history(period="max")
    nasdaq.to_csv("nasdaq.csv")

# Ensure the index is in datetime format and in UTC
nasdaq.index = pd.to_datetime(nasdaq.index, utc=True)

# Remove irrelevant columns and prepare target
del nasdaq["Dividends"]
del nasdaq["Stock Splits"]
nasdaq["Tomorrow"] = nasdaq["Close"].shift(-1)
nasdaq["Target"] = (nasdaq["Tomorrow"] > nasdaq["Close"]).astype(int)

# Filter data from 2003 onwards
nasdaq = nasdaq.loc["2003-01-01":].copy()

# Define predictors for the initial model
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train and evaluate the initial model
initial_model = RandomForestClassifier(
    n_estimators=100, min_samples_split=100, random_state=1)
train = nasdaq.iloc[:-100]
test = nasdaq.iloc[-100:]
initial_model.fit(train[predictors], train["Target"])
initial_preds = initial_model.predict(test[predictors])
initial_precision = precision_score(test["Target"], initial_preds)
print("Precision Score (Initial Model):", initial_precision)

# Define prediction function


def predict(train, test, predictors, model, threshold=0.55):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= threshold).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Define backtesting function


def backtest(data, model, predictors, start=2500, step=250, threshold=0.55):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model, threshold)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# Perform backtesting with the initial model
predictions = backtest(nasdaq, initial_model, predictors)
print("\nBacktested Initial Model Results:")
print("Predictions Counts:", predictions["Predictions"].value_counts())
print("Precision Score (Backtested Initial Model):", precision_score(
    predictions["Target"], predictions["Predictions"]))
print("Target Distribution (Initial Model):\n",
      predictions["Target"].value_counts() / predictions.shape[0])

# Define and train the enhanced model with a higher threshold and additional features
enhanced_model = RandomForestClassifier(
    n_estimators=200, min_samples_split=50, random_state=1)
horizons = [2, 5, 20]
new_predictors = []

for horizon in horizons:
    rolling_averages = nasdaq.rolling(horizon).mean()
    nasdaq[f"Close_Ratio_{horizon}"] = nasdaq["Close"] / \
        rolling_averages["Close"]
    nasdaq[f"Trend_{horizon}"] = nasdaq.shift(
        1).rolling(horizon).sum()["Target"]
    new_predictors += [f"Close_Ratio_{horizon}", f"Trend_{horizon}"]

# Drop rows with NaN values after feature engineering
nasdaq.dropna(inplace=True)

# Combine initial and new predictors for the enhanced model
all_predictors = predictors + new_predictors

# Perform backtesting with the enhanced model
predictions = backtest(nasdaq, enhanced_model, all_predictors, threshold=0.6)
enhanced_precision = precision_score(
    predictions["Target"], predictions["Predictions"])
print("\nEnhanced Model Results:")
print("Precision Score (Enhanced Model):", enhanced_precision)
# Display value counts for predictions and target distribution
print("Predictions Counts:\n", predictions["Predictions"].value_counts())
print("Target Distribution:\n", predictions["Target"].value_counts() / predictions.shape[0])

# Add a final interpretation of the model's prediction accuracy
if enhanced_precision > 0.7:
    print("This means the enhanced model has a high likelihood of correctly predicting if the Nasdaq will go up tomorrow based on the features used.")
elif enhanced_precision > 0.5:
    print("This means the enhanced model performs moderately well and has some predictive value for forecasting if the Nasdaq will rise tomorrow, but it may not be consistently reliable.")
else:
    print("This means the enhanced model's predictions are close to random guessing for whether the Nasdaq will go up tomorrow, and further tuning or additional features may be required.")
