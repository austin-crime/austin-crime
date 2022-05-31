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

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

# Scale data after splitting with MinMax
def scale_data(train, validate, test, columns_to_scale, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    '''
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def visualize_scaler(scaler, df, target_columns, bins=10):
    fig, axs = plt.subplots(len(target_columns), 2, figsize=(15, 12))
    df_scaled = df.copy()
    df_scaled[target_columns] = scaler.fit_transform(df[target_columns])
    for (ax1, ax2), col in zip(axs, target_columns):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.98, 
                        wspace=0.4, 
                        hspace=0.4)
    plt.tight_layout()
    return fig, axs


######### Cleaning Steps ##############

def prep_data(df):
    
    # Change occurence date to datetime type in order to subeset data
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
        'occ_date_time',
        'rep_date_time']
    
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
    'occ_date' : 'occurence_date',
    'occ_time' : 'occurence_time',
    'rep_date' : 'report_date',
    'rep_time' : 'report_time'}
    
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
    
    ## Adding the trailing zeros for military time 
    df.occurence_time = df.occurence_time.apply(lambda time: f'{int(time):04d}')
    df.report_time = df.report_time.apply(lambda time: f'{int(time):04d}')
    
    ## Converting to datetime format

    df.report_date = pd.to_datetime(df.report_date, format = '%Y-%m-%d')
    df.clearance_date = pd.to_datetime(df.clearance_date, format = '%Y-%m-%d')
    df.occurence_time = pd.to_datetime(df.occurence_time, format = '%H%M')
    df.report_time = pd.to_datetime(df.report_time, format = '%H%M')

    df.occurence_time = df.occurence_time.dt.strftime('%H:%M')
    df.report_time = df.report_time.dt.strftime('%H:%M')

    # Create new target variable with True or False values where "not cleared" 
    # is False and "cleared by arrest" and "cleared by exception" are True.
    clearance = np.where(df.clearance_status == 'not cleared', False, True)
    df['cleared'] = clearance
    df.cleared.value_counts()
    
    return df 

    


