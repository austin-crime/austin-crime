import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def attribute_nulls(df):
    '''
    This function takes in a dataframe and checking all the rows if they have any missing values
    and return a dataframe with the number of missing values and percent missing.
    '''
    nulls = df.isnull().sum()
    rows = len(df)
    percent_missing = nulls / rows 
    dataframe = pd.DataFrame({'rows_missing': nulls, 'percent_missing': percent_missing})
    return dataframe

def column_nulls(df):
    '''
    This function takes in a dataframe and checking all the columns if they have any missing values
    and return a dataframe with with the number of missing value and percent missing.
    '''
    new_df = pd.DataFrame(df.isnull().sum(axis=1), columns = ['cols_missing']).reset_index()\
    .groupby('cols_missing').count().reset_index().\
    rename(columns = {'index': 'rows'})
    new_df['percent_missing'] = new_df.cols_missing/df.shape[1]
    return new_df
    
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    '''
    This function takes in a dataframe and then drop the missing values.
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def split_data(df):

    """
    This function takes in a dataframe, then splits and returns the data as train, validate, and test sets 
    using random state 42.
    """
    # split data into 2 groups, train_validate and test, assigning test as 20% of the dataset
    train_validate, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["cleared"]
    )
    # split train_validate into 2 groups with
    train, validate = train_test_split(
        train_validate,
        test_size=0.3,
        random_state=42,
        stratify=train_validate["cleared"],
    )
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

def prep_data(df):
    
    # Change occurrence date to datetime type in order to subeset data
    df.occ_date = pd.to_datetime(df.occ_date, format = '%Y-%m-%d')
    
    # Subset the data to include observations between 2018-01-01 and 2021-12-31.
    df = df[(df.occ_date >= '2018-01-01') & (df.occ_date <= '2021-12-31')]
    
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
    
    # Here we'll fill missing values for some columns with a value we have decided upon.
    df['location_type'] = df.location_type.fillna('OTHER / UNKNOWN') #filling all na with the Other/Unknown value
    df['council_district'] = df.council_district.fillna(9) # filling all na with the most common district
    
    # Renaming columns for clarity and readability
    
    mapper = {
    'occ_date' : 'occurrence_date',
    'occ_date_time' : 'occurrence_time',
    'rep_date' : 'report_date',
    'rep_date_time' : 'report_time'}
    
    df = df.rename(columns = mapper)
    
    # Changing clearance status values to a more readable version using the data documentation 
    
    clearance_mapper = {
    'N' : 'not cleared',
    'O' : 'cleared by exception',
    'C' : 'cleared by arrest'}
    
    df['clearance_status'] = df.clearance_status.map(clearance_mapper)
    
    # Changing data to numeric types where appropriate
    df.latitude = df.latitude.astype('float')
    df.longitude = df.longitude.astype('float')

    # We want to change the date and time columns to datetime types.
    
    ## Converting to datetime format

    df.report_date = pd.to_datetime(df.report_date, format = '%Y-%m-%d')
    df.clearance_date = pd.to_datetime(df.clearance_date, format = '%Y-%m-%d')
    df.occurrence_time = pd.to_datetime(df.occurrence_time, format = '%Y-%m-%dT%H:%M:%S')
    df.report_time = pd.to_datetime(df.report_time, format = '%Y-%m-%dT%H:%M:%S')

    # Create new target variable with True or False values where "not cleared" 
    # is False and "cleared by arrest" and "cleared by exception" are True.
    clearance = np.where(df.clearance_status == 'not cleared', False, True)
    df['cleared'] = clearance
    df.cleared.value_counts()
    
    return df 

    


