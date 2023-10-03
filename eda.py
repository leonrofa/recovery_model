import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# function definition
def find_outliers(column):
    """identify outliers as defined by the interquartile range without considering missing values"""
    column_clean = column.dropna()
    q1 = column_clean.quantile(0.25)
    q3 = column_clean.quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    return df[(column_clean < lb) | (column_clean > ub)]

# load dataframe
df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.rename(columns={'icu_rpe':'rpe'}, inplace=True)

# check missing values and duplicates
print(df.isnull().sum())
print(df[df.duplicated()])

# identify outliers using iqr
outliers = {col: find_outliers(df[col]) for col in df.select_dtypes(include=['number'])}
for col, outlier_df in outliers.items():
    if not outlier_df.empty:
        print(f"Outliers in column {col}:\n", outlier_df)

# plot 'feel' time series
plt.figure(figsize=(10,2))
plt.plot(df['feel'])
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/feel_original.pdf', format='pdf')
plt.show()

# process and plot 'feel' time series by day of the week
df['week_year'] = df.index.isocalendar().week.astype(str) + '_' + df.index.year.astype(str)
df['dayweek'] = df.index.weekday

plt.figure(figsize=(10,6))
for week_year in df['week_year'].unique():
    week_data = df[df['week_year'] == week_year].sort_values(by='dayweek')
    plt.plot(week_data['dayweek'], week_data['feel'], marker='o', markerfacecolor='none', color='C0', alpha=0.25)

plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel('Day of the week')
plt.ylabel('Feel (1: strong––5: weak)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/seasonal_effect.pdf', format='pdf')
plt.show()

# plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
feel_clean = df['feel'].dropna()
plot_acf(feel_clean, ax=ax1, lags=50)
plot_pacf(feel_clean, ax=ax2, lags=50)
plt.tight_layout()
plt.savefig('figures/correlations.pdf', format='pdf')
plt.show()