import requests
import pandas as pd
from config import API_KEY

# base url and endpoint details
BASE_URL = 'https://intervals.icu/api/v1/athlete/i49168/activities/'
PARAMS = {
    'oldest': '2022-02-06',
    'newest': '2023-09-27'
}

# send GET request to the api
response = requests.get(BASE_URL, params=PARAMS, auth=('API_KEY', API_KEY))

# check for successful response
if response.status_code == 200:
    # convert the json response to a pandas dataframe
    df = pd.DataFrame(response.json())

    # display basic dataframe info
    print(df.shape)
    print(df.head())

    # save the dataframe to a csv file
    df.to_csv('activities.csv', index=False)
else:
    print(f"failed to fetch data. HTTP status code: {response.status_code}")
