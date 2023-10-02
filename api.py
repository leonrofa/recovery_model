import requests
from requests.auth import HTTPBasicAuth
import json
import pandas as pd

from config import API_KEY

response = requests.get(f'https://intervals.icu/api/v1/athlete/i49168/activities/?oldest=2022-02-06&newest=2023-09-27', auth=HTTPBasicAuth('API_KEY', API_KEY))
print(response.status_code)
data = response.json()

df = pd.DataFrame(data)
df.shape
df.head()
df.to_csv('activities.csv', index=False)

# response2 = requests.get(f'https://intervals.icu/api/v1/activity/{API_KEY}/power-histogram/?id=i24340942', auth=HTTPBasicAuth(USERNAME, API_KEY))
# print(response.status_code)
# power = response.json()
# df2 = pd.DataFrame(power)
# for col in df2.columns:
#     print(col)