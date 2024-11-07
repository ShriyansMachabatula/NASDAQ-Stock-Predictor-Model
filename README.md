# NASDAQ-Stock-Predictor-Model
 A machine learning model predicting the daily movement of the NASDAQ Composite index. Using historical data and Random Forest Classifier, it classifies if the index will rise or fall the next day. Enhanced with trend and ratio features, backtested for precision, and includes interpretation of prediction accuracy for realistic insights.

1. At the start of the script, the program checks if the historical data for NASDAQ exists locally as nasdaq.csv. There is an if statement which will obtain the nasdaq.csv file. We use an if statement to reduce API calls. The date index is converted to UTC to handle timezone differences, ensuring consistent date-based filtering. The target variable, Target, is set to 1 if the following day’s price is higher than today’s (Tomorrow > Close), and 0 otherwise.


2. The initial model uses key financial indicators (Close, Volume, Open, High, Low) as predictors. Using data from 2003 onward, it reserves the last 100 entries for testing, with the rest is used for training. A basic RandomForestClassifier trains on these predictors to provide a baseline accuracy (precision score) for the model’s predictions.


3. The predict() function trains the model and makes predictions based on a set threshold. backtest() iteratively simulates real-world conditions by training on historical data chunks and testing on the following periods, evaluating how the model performs over time.


4. In order to improve accuracy we try and add more parameters to factor in when classifying. In this case we use a Close_ratio which is the measure of the ratio of today’s closing price to its rolling average over specific horizons (e.g., 2, 5, 20 days), capturing price momentum. Another case is when we measure Trend which sums recent target values over each horizon, identifying the frequency of price increases to reflect trends.


5. When new factors are added, the model is retrained with a higher level (for example, 0.6) to focus on more accurate predictions and cut down on false positives. We then backtest the improved model on the whole dataset gives us better results and a new accuracy score that show how well it works with both short- and medium-term trends.


6. The model’s effectiveness is measured by precision scores, prediction counts, and target distribution. A high accuracy score means that the model is reliable, while a middling score means that it can make some predictions. According to our results the model performs moderately well and has some predictive value for forecasting if the Nasdaq will rise tomorrow, but it may not be consistently reliable.
   
