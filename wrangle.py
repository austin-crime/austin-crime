'''

    wrangle.py

    Description: Contains functions that can be used for both acquiring and preparing 
        the Austin crime data in a single step.

'''

################################################################################

from acquire import get_crime_data
from prepare import prep_data
import pandas as pd

################################################################################

def wrangle_crime_data(drop_cleared_by_exception = False):
    '''
        Acquire and prepare the Austin crime data from data.austintexas.gov.
    '''

    return prep_data(get_crime_data(), drop_exception = drop_cleared_by_exception)




def wrangle_merged_df():
    df = wrangle_crime_data(drop_cleared_by_exception=True)
    # Acquire our two dfs to be merged
    main_df = df.copy()
    districts_df = pd.read_csv('Austin_Police_Department_Districts_data.csv')
    
    # Clean up the columns in districts_df to match format of main_df
    districts_df.columns = districts_df.columns.str.strip().str.lower().str.replace('__', ('_'))
    
    # Change the values in sectors to more closely match those in distrcits_df
    sectors = {'ED':'EDWARD', 'DA':'DAVID', 'FR':'FRANK', 'AD':'ADAM', 
               'BA':'BAKER', 'CH':'CHARLIE', 'HE':'HENRY', 'ID':'IDA',
               'GE':'GEORGE', 'AP': 'APT', '88':'88', 'UT':'UT'}

    main_df['sector'] = main_df.sector.map(sectors)
    
    # Cobine the two columns into one called pd_district to match column in other data set
    main_df['pd_district'] = main_df['sector'].astype(str) + ' ' + df['district'].astype(str)
    
    # Merge the data sets using the newly created column in main_df
    merged = main_df.merge(districts_df, left_on= ['pd_district'], right_on= ['district_name'])
    
    return merged 
    # Acquire our two dfs to be merged
    main_df = df.copy()
    districts_df = pd.read_csv('Austin_Police_Department_Districts_data.csv')
    
    # Clean up the columns in districts_df to match format of main_df
    districts_df.columns = districts_df.columns.str.strip().str.lower().str.replace('__', ('_'))
    
    # Change the values in sectors to more closely match those in distrcits_df
    sectors = {'ED':'EDWARD', 'DA':'DAVID', 'FR':'FRANK', 'AD':'ADAM', 
               'BA':'BAKER', 'CH':'CHARLIE', 'HE':'HENRY', 'ID':'IDA',
               'GE':'GEORGE', 'AP': 'APT', '88':'88', 'UT':'UT'}

    main_df['sector'] = main_df.sector.map(sectors)
    
    # Cobine the two columns into one called pd_district to match column in other data set
    main_df['pd_district'] = main_df['sector'].astype(str) + ' ' + df['district'].astype(str)
    
    # Merge the data sets using the newly created column in main_df
    merged = main_df.merge(districts_df, left_on= ['pd_district'], right_on= ['district_name'])
    
    return merged 