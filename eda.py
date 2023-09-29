import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# import dataframe
df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.rename(columns={'icu_rpe':'rpe'}, inplace=True)
df.head()

# check na's, duplicates
print(df.isnull().sum())
print(df[df.duplicated()])

# identify outliers as defined by iqr 
numeric_cols = df.select_dtypes(include=['number'])

outliers = {}

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lb = q1-1.5*iqr
    ub = q3+1.5*iqr

    outliers[col] = df[(df[col] < lb) | (df[col] > ub)]

for col, outlier_df in outliers.items():
    if not outlier_df.empty:
        print(f"Outliers in column {col}:\n", outlier_df)

# plot feel
plt.figure(figsize=(10,2))
plt.plot(df.index, df[['feel']])
plt.grid(True)
plt.tight_layout()
plt.show()

# no trend, looks stationary

# plot feel by day of week 
df['week_year'] = df.index.isocalendar().week.astype(str) + '_' + df.index.year.astype(str)
df['dayweek'] = df.index.weekday

plt.figure(figsize=(10,6))
for week_year in df['week_year'].unique():
    week_data = df[df['week_year'] == week_year].sort_values(by='dayweek')
    plt.plot(week_data['dayweek'], week_data['feel'], marker='o', markerfacecolor='none', color='C0', alpha=0.25)

plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel('Day of the week')
plt.ylabel('feel (1: strong––5: weak)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# no clear seasonal effect

# plot sample acf of feel
fig, ax = plt.subplots(figsize=(10,6))
plot_acf(df['feel'], ax=ax, lags=50)
plt.show()

# some persistency in the time series

# box plots of features
features = ['feel', 'moving_time', 'rpe']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,6))
for ax, feature in zip(axes, features):
    ax.boxplot(df[feature], vert=True)
    ax.set_title(feature)
    ax.set_xticks([])

fig.suptitle('Box-and-whisker plots of features (note: non-iid obs)')

plt.tight_layout()
plt.show()