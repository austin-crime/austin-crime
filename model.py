'''

    model.py

    Description: This file contains functions used for producing machine learning 
        models for the Austin Crime project.

    Variables:

        random_seed

    Classes:

        Model(self, model, train, features, target)

    Functions:

        prep_data_for_modeling(df)
        train_models(train)

'''

################################################################################

import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB

################################################################################

random_seed = 42

################################################################################

class Model:

    '''
        A Model class which can be used to easily keep track of the algorithm, 
        feature set, and target variable a machine learning model utilizes.
    
        Instance Fields
        ---------------
        model: sklearn model
        train: DataFrame
        features: list[str]
        target: str
    
        Instance Methods
        ----------------
        __init__: Returns None
        make_preditions: Returns numpy.array
    '''

    ################################################################################

    def __init__(self, model, train: pd.DataFrame, features: list[str], target: str) -> None:
        '''
            Initialize the Model class with the given model object and dataframe.
        
            Parameters
            ----------
            model: sklearn model
                A sklearn machine learning model to use for making predictions 
                on a classification or regression problem.
        
            train: DataFrame
                A pandas dataframe containing the training data that will be 
                used to fit the sklearn model.

            features: list[str]
                A list of features in train to use when fitting the sklearn 
                model and when making predictions.

            target: str
                The target variable in train to use when fitting the sklearn
                model and when making predictions.
        '''

        self.model = model
        self.features = features
        self.target = target

        self.model.fit(train[self.features], train[self.target])

    ################################################################################

    def make_predictions(self, df: pd.DataFrame) -> np.array:
        '''
            Make predictions of the target variable using the provided 
            dataframe.
        
            Parameters
            ----------
            df: DataFrame
                A pandas dataframe with which to make predictions of the 
                target variable.
        
            Returns
            -------
            array: A numpy array containing the predictions of the target 
                variable.
        '''

        return self.model.predict(df[self.features])    

################################################################################

def prep_data_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Prepare the data for modeling by removing features that won't be used, and 
        encoding features as types that the machine learning models will be able to 
        use.
    '''

    # These are the features that will be used in modeling.
    features = [
        'crime_type',
        'council_district',
        'latitude',
        'longitude',
        'cleared',
        'time_to_report'
    ]

    df = df[features]

    # One hot encode the crime_type feature and drop the crime_type feature.
    dummy_df = pd.get_dummies(df['crime_type'])
    df = pd.concat([df, dummy_df], axis = 1)
    df = df.drop(columns = 'crime_type')

    # Convert the Timedelta type in time_to_report to a float representing the time in seconds.
    df['time_to_report'] = df.time_to_report / np.timedelta64(1, 's')

    return df

################################################################################

def train_models(train: pd.DataFrame) -> dict:
    '''
        Build and train machine learning models for the final report.
    '''

    algorithms = {
        'Ada Boost' : AdaBoostClassifier(random_state = random_seed),
        'Bagging Classifier' : BaggingClassifier(random_state = random_seed),
        'Naive Bayes' : BernoulliNB()
    }

    models = {}

    for key, algorithm in algorithms.items():
        models[key] = Model(
            algorithm,
            train = train,
            features = train.drop(columns = 'cleared').columns,
            target = 'cleared'
        )

    return models