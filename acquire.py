import pandas as pd
import os
from sodapy import socrata

csv = 'Crime_Reports.csv'

def get_crime_data(use_cache=True):
    '''This function returns the data from the Crime Reports csv. 
        
    '''
    if os.path.exists(csv) and use_cache:
        print('Using cached csv')
        return pd.read_csv(csv)
    print('Acquiring data from api')

    df = read_from_api()
    df.to_csv(csv)
    
    return df

def read_from_api():
    client = Socrata("data.austintexas.gov", None)
    results = client.get("fdj4-gpfu", limit=500000)
    results_df = pd.DataFrame.from_records(results)
    return results_df