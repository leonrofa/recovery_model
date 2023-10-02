import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import q_stat
from statsmodels.tsa.api import acf
import matplotlib.pyplot as plt
import numpy as np

# import dataframe
df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.rename(columns={'icu_rpe':'rpe'}, inplace=True)
df.head()

# ADF test, Ljung-Box test
result_adf = adfuller(df['feel'])
print('# of lags:', result_adf[2])
print('p-value:', result_adf[1])

# reject that there's a unit root, conclude time series is stationary

acf_vals = acf(df['feel'])
result_lb = q_stat(acf_vals, len(df['feel']))
print('p-value:', result_lb[1][-1])

# reject that the time series is white noise (iid)

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
def compute_rmse(errors):
    """compute the root mean square error"""
    return np.sqrt(np.mean(np.square(errors)))

def expanding_window_forecast(y, start_point, horizon=1):
    """compute expanding-window residuals of h-step ahead predictions"""
    errors = []
    for i in range(start_point, len(y)-horizon):
        model = sm.tsa.ARIMA(y.iloc(:i+1), order=(1,0,1)).fit()
        predicted_value = model.forecast(steps=horizon)[0][-1]
        errors.append(y.iloc[i+horizon]-predicted_value)
    return errors

y = pd.Series(df['feel'])

start_point = round(0.75*len(y))
errors = expanding_window_forecast(y, start_point)
rmse = compute_rmse(errors)

print(rmse)
