import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import numpy as np

# import dataframe
df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.rename(columns={'icu_rpe':'rpe'}, inplace=True)
df.head()

# model 1: linear least squares model
X = df[['moving_time', 'rpe']]
y = df['feel']

m1 = LinearRegression()
m1.fit(X, y)
print(m1.coef_); print(m1.intercept_)
print(m1.score(X, y))

# model 2: ARMA(1,1) 
m2 = sm.tsa.ARIMA(df['feel'], order=(1,0,1))
m2_result = m2.fit()

# plot residuals for model diagnostics
fig = m2_result.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()

# some deviations from normality in the tails; no significant autocorrelations

# out-of-sample model evalution with expanding-window approach and RMSE 
train_size = int(len(df['feel']) * 0.7)
train_data = df['feel'][:train_size].tolist()
test_data = df['feel'][train_size:].tolist()

predictions = []

for i in range(0, len(test_data), 5):
    combined_data = train_data[i:]
    model = sm.tsa.ARIMA(combined_data, order=(1,0,1))
    model_fit = model.fit()
    
    steps_to_forecast = min(5, len(test_data)-i)
    predicted = model_fit.forecast(steps=steps_to_forecast)[0]

    if not isinstance(predicted, (list, np.ndarray)):
        predicted = [predicted]

    predictions.extend(predicted)

m2_rmse = np.sqrt(mean_squared_error(test_data, predictions))
print(f"Root Mean Squared Error (RMSE): {m2_rmse:.2f}")

