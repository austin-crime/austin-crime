import pandas as pd
import os
from sodapy import Socrata

# The API documentation states that a user should ideally create an app token to use the API,
# however, it is not required and for the purposes of this project not having an app token 
# does not cause any issues.

# Attempt to import the app_token from env.py, if that fails set app_token to None.
try:
    from env import app_token
except ImportError as e:
    app_token = None

# Give a global variable name for csv
csv = 'Crime_Reports.csv'

def get_crime_data(use_cache=True):
    '''
    This function returns the data from the Crime Reports csv.  
    '''
    # Checks if cache exists
    if os.path.exists(csv) and use_cache:
        print('Using cached csv')
        # Read data from csv into Pandas Dataframe
        return pd.read_csv(csv)
    print('Acquiring data from api')
    # Returned as JSON from API / converted to Python list of
    # dictionaries by sodapy.
    # Convert to pandas DataFrame
    df = read_from_api()
    # Convert to csv
    df.to_csv(csv, index = False)
    
    return df

def read_from_api():
    '''
    First 500,000 results, returned as JSON from API / converted to Python list of
    dictionaries by sodapy.
    '''
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    # client = Socrata("data.austintexas.gov", None)
    
    # Example authenticated client (needed for non-public datasets):
    # client = Socrata(data.austintexas.gov,
    #                  MyAppToken,
    #                  userame="user@example.com",
    #                  password="AFakePassword")
    
    # A limited number of requests can be made without an app token, 
    # but they are subject to much lower throttling limits than request that do include one.
    # With an app token, your application is guaranteed access to it's own pool of requests
    client = Socrata("data.austintexas.gov", app_token)
    
    # .get(dataset_identifier) 
    # dataset_identifier: is a part of url, usually appear near the end of an url, 
    # appear as 8 letter and number combinations with a hyphen 
    results = client.get("fdj4-gpfu", limit=500_000)
    
    results_df = pd.DataFrame.from_records(results)
    return results_df