import pandas as pd

# import dataframe
df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.rename(columns={'icu_rpe':'rpe'}, inplace=True)
df.head()