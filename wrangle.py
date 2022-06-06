'''

    wrangle.py

    Description: Contains functions that can be used for both acquiring and preparing 
        the Austin crime data in a single step.

'''

################################################################################

from acquire import get_crime_data
from prepare import prep_data

################################################################################

def wrangle_crime_data(drop_cleared_by_exception = False):
    '''
        Acquire and prepare the Austin crime data from data.austintexas.gov.
    '''

    return prep_data(get_crime_data(), drop_exception = drop_cleared_by_exception)