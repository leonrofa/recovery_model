import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, q_stat, acf

# function definitions
def compute_rmse(errors):
    """compute the root mean square error"""
    return np.sqrt(np.mean(np.square(errors)))

def expanding_window_forecast(y, start_point):
    """compute expanding-window residuals of 1-step ahead predictions"""
    errors = []
    for i in range(start_point, len(y)-1):
        model = sm.tsa.ARIMA(y.iloc[:i+1], order=(1,0,1)).fit()
        predicted_value = model.forecast(steps=1)[0]
        errors.append(y.iloc[i+1]-predicted_value)
    return errors

# load dataframe
df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.rename(columns={'icu_rpe':'rpe'}, inplace=True)

# filter and process data
univariate_df = df.loc[df['type'].isin(['Elliptical', 'Run']), 'feel']
univariate_df = univariate_df.resample('D').mean().interpolate(method='linear')

# plot time series
plt.figure(figsize=(10,2))
plt.plot(univariate_df)
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/feel_processed.pdf', format='pdf')
plt.show()

# adf and ljung-box tests
result_adf = adfuller(univariate_df)
print('# of lags:', result_adf[2])
print('p-value:', result_adf[1])

# reject that there's a unit root, conclude time series is stationary

acf_vals = acf(univariate_df)
result_lb = q_stat(acf_vals, len(df['feel']))
print('p-value:', result_lb[1][-1])

# reject that the time series is white noise (iid)

# autoregressive model for univariate time series
m1 = sm.tsa.ARIMA(univariate_df, order=(1,0,1))
m1_result = m1.fit()

# plot residuals for model diagnostics
fig = m1_result.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()

# some deviations from normality in the tails; no significant autocorrelations

# out-of-sample model evaluation with expanding-window approach and RMSE
start_point = round(0.7*len(univariate_df))
errors = expanding_window_forecast(univariate_df, start_point)
rmse = compute_rmse(errors)
print(round(rmse, 4))
