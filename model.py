import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv('activities.csv', parse_dates=['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.head()

plt.figure(figsize=(10,6))
plt.plot(df.index, df[['icu_rpe']])
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(df.index, df[['icu_rpe']])
plt.grid(True)
plt.tight_layout()
plt.show()
