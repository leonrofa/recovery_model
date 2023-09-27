import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.rename(columns={'icu_rpe':'rpe'}, inplace=True)
df.head()

# plot feel
plt.figure(figsize=(10,1))
plt.plot(df.index, df[['feel']])
plt.grid(True)
plt.tight_layout()
plt.show()

# plot sample acf of feel
fig, ax = plt.subplots(figsize=(10,6))
plot_acf(df['feel'], ax=ax, lags=50)
plt.show()

# seasonal plot by weekday, month
df['weekday'] = df.index.dayofweek
df['month'] = df.index.isocalendar().week

grouped_weekday = [df['feel'][df['weekday'] == day].values for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
grouped_month = [df['feel'][df['month'] == month].values for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
ax1.boxplot(grouped_weekday, vert=True, patch_artist=True, labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
ax1.set_xlabel('Weekday')
ax2.boxplot(grouped_month, vert=True, patch_artist=True, labels=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
plt.tight_layout()
plt.show()