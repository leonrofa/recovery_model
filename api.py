import requests
from requests.auth import HTTPBasicAuth
import json
import pandas as pd

from config import USERNAME, API_KEY

response = requests.get(f'https://intervals.icu/api/v1/athlete/{API_KEY}/activities/?oldest=2022-02-06&newest=2023-09-27', auth=HTTPBasicAuth(USERNAME, API_KEY))
print(response.status_code)
data = response.json()

df = pd.DataFrame(data)
df = df[['id', 'type', 'start_date_local', 'moving_time', 'icu_rpe', 'feel']]
df.dropna(inplace=True)
df.shape
df.head()
df.to_csv('activities.csv', index=False)

response2 = requests.get(f'https://intervals.icu//api/v1/activity/{API_KEY}/fit-file/?oldest=2023-09-26&newest=2023-09-27', auth=HTTPBasicAuth(USERNAME, API_KEY))
print(response.status_code)
fitfile = response.json()
df2 = pd.DataFrame(fitfile)
for col in df2.columns:
    print(col)

