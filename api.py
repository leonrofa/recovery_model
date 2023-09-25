import requests
from requests.auth import HTTPBasicAuth
import json
import pandas as pd

from config import USERNAME, API_KEY

response = requests.get('https://intervals.icu/api/v1/athlete/i49168/activities.csv', auth=HTTPBasicAuth(USERNAME,API_KEY))
print(response.status_code)

filename = 'activities.csv'
with open(filename, 'w') as file:
    file.write(response.text)

activity_ids = pd.read_csv('/Users/leonrofagha/recovery_model/activities.csv', usecols=['id'])
print(activity_ids.head())