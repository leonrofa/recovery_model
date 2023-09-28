import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# import dataframe
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

# plot feel by day of week 
df['week_year'] = df.index.isocalendar().week.astype(str) + '_' + df.index.year.astype(str)
df['dayweek'] = df.index.weekday

plt.figure(figsize=(10,6))
for week_year in df['week_year'].unique():
    week_data = df[df['week_year'] == week_year].sort_values(by='dayweek')
    plt.plot(week_data['dayweek'], week_data['feel'], grouped.loc[week_year], marker='o', markerfacecolor='none', color='C0', alpha=0.25)

plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel('Day of the week')
plt.ylabel('Feel (1: strong––5: weak)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# no clear weekly effect