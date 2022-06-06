import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def attribute_nulls(df):
    nulls = df.isnull().sum()
    rows = len(df)
    percent_missing = nulls / rows 
    dataframe = pd.DataFrame({'rows_missing': nulls, 'percent_missing': percent_missing})
    return dataframe

def column_nulls(df):
    new_df = pd.DataFrame(df.isnull().sum(axis=1), columns = ['cols_missing']).reset_index()\
    .groupby('cols_missing').count().reset_index().\
    rename(columns = {'index': 'rows'})
    new_df['percent_missing'] = new_df.cols_missing/df.shape[1]
    return new_df
    
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


#Split the data into train, validate and test to ensure the data not leakage 
def split_data(df, target = 'cleared', seed = 123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

#remove outliers
def get_lower_and_upper_bounds(df,col, k=1.5):
    '''
    takes in a df, a column from the df, and a multiplier (default is 1.5)
    calculates the IQR, prints out the lower and upper bound, and returns entries 
    below the lower bound, and higher than the upper bound
    '''
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1 # calculate interquartile range
    upper_value = q3 + (k * iqr) # get upper bound
    lower_value = q1 - (k * iqr) # get lower bound
    #return dataframe without outliers
    under_bound = df[df[col] < lower_value]
    over_bound = df[df[col] > upper_value]
    print(f'for {col}, the lower bound is {lower_value}, and the upper bound is {upper_value}')
    return under_bound, over_bound


######### Cleaning Steps ##############

def prep_data(df, drop_exception = False):
    
    df = reduce_time_frame(df)
    df = drop_columns(df)
    df = remove_rows_missing_values(df)
    df = impute_missing_values(df)
    df = rename_columns(df)
    df = rename_values(df)
    df = cast_column_types(df)
    if drop_exception: df = drop_cleared_by_exception(df)
    df = engineer_features(df)
    
    return df 

################################################################################

def reduce_time_frame(df: pd.DataFrame) -> pd.DataFrame:

    # Change occurrence date to datetime type in order to subeset data
    df.occ_date = pd.to_datetime(df.occ_date, format = '%Y-%m-%d')
    
    # Subset the data to include observations between 2018-01-01 and 2021-12-31.
    df = df[(df.occ_date >= '2018-01-01') & (df.occ_date <= '2021-12-31')]

    return df

################################################################################

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:

    # These are all the columns that will be dropped from the dataframe.
    columns = [
        'incident_report_number',
        'ucr_code',
        'ucr_category',
        'category_description',
        ':@computed_region_a3it_2a2z',
        ':@computed_region_8spj_utxs',
        ':@computed_region_q9nd_rr82',
        ':@computed_region_qwte_z96m',
        'x_coordinate',
        'y_coordinate',
        'location',
        'census_tract',
        'pra',
        'occ_time',
        'rep_time']
    
    # Drop duplicated information and unncessary/unuseful columns
    df = df.drop(columns = columns)

    return df

################################################################################

def remove_rows_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    # Here we'll drop rows with missing values that cannot be reasonabled imputed with a value.
    null_columns = [
    'clearance_status',
    'clearance_date',
    'zip_code',
    'sector',
    'district',
    'latitude',
    'longitude']
    
    for column in null_columns:
        df = df[~df[column].isna()]

    return df

################################################################################

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    # Here we'll fill missing values for some columns with a value we have decided upon.
    df['location_type'] = df.location_type.fillna('OTHER / UNKNOWN') #filling all na with the Other/Unknown value
    df['council_district'] = df.council_district.fillna(9) # filling all na with the most common district

    return df

################################################################################

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:

    # Renaming columns for clarity and readability
    
    mapper = {
    'occ_date' : 'occurrence_date',
    'occ_date_time' : 'occurrence_time',
    'rep_date' : 'report_date',
    'rep_date_time' : 'report_time'}
    
    df = df.rename(columns = mapper)

    return df

################################################################################

def rename_values(df: pd.DataFrame) -> pd.DataFrame:

    # Changing clearance status values to a more readable version using the data documentation 
    
    clearance_mapper = {
    'N' : 'not cleared',
    'O' : 'cleared by exception',
    'C' : 'cleared by arrest'}
    
    df['clearance_status'] = df.clearance_status.map(clearance_mapper)

    return df

################################################################################

def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:

    # Changing data to numeric types where appropriate
    df.latitude = df.latitude.astype('float')
    df.longitude = df.longitude.astype('float')

    # We want to change the date and time columns to datetime types.
    
    ## Converting to datetime format

    df.report_date = pd.to_datetime(df.report_date, format = '%Y-%m-%d')
    df.clearance_date = pd.to_datetime(df.clearance_date, format = '%Y-%m-%d')
    df.occurrence_time = pd.to_datetime(df.occurrence_time, format = '%Y-%m-%dT%H:%M:%S')
    df.report_time = pd.to_datetime(df.report_time, format = '%Y-%m-%dT%H:%M:%S')

    return df

################################################################################

def drop_cleared_by_exception(df: pd.DataFrame) -> pd.DataFrame:

    df = df[~(df.clearance_status == 'cleared by exception')]
    return df

################################################################################

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Create new target variable with True or False values where "not cleared" 
    # is False and "cleared by arrest" and "cleared by exception" are True.
    clearance = np.where(df.clearance_status == 'not cleared', False, True)
    df['cleared'] = clearance

    # Create a feature that is the difference between when a crime occurred and 
    # when it was reported.
    df['time_to_report'] = df.report_time - df.occurrence_time

    return df

################################################################################

def merge_districts(df):
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
    