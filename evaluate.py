'''

    baseline.py

    Description: This file provides functions for easily establishing baseline 
        models in machine learning problems.

    Functions:

        establish_classification_baseline(target)
        append_model_results(index, results, evaluate_df = None)

'''

import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score

################################################################################

def establish_classification_baseline(target: pd.DataFrame) -> pd.Series:
    '''
        Returns a pandas series containing the most common value in the target 
        variable that is of the same size as the provided target. This series 
        serves as the baseline model to which to compare any machine learning 
        models.

        Parameters
        ----------
        target: DataFrame
            A pandas series containing the target variable for a machine 
            learning project.

        Returns
        -------
        Series: A pandas series with the same size as target filled with the 
            most common value in target.
    '''

    most_common_value = target.mode()[0]
    return pd.Series(most_common_value, index = target.index)

################################################################################

def append_model_results(name: str, results: dict, evaluate_df: pd.DataFrame = None) -> pd.DataFrame:
    '''
        Append the evaluation results to the evaluate_df or if an evaluate_df 
        is not provided, create one and append the results.
    
        Parameters
        ----------
        name: str
            The name to assign to the results entry provided. A string provides 
            a more descriptive name, but any valid dataframe index is acceptable.
        results: dict[str : float]
            The results of the model evaluation in the form of a dictionary with 
            the metric as the key and the result as a float.
        evaluate_df: DataFrame, optional
            The evaluation dataframe to append the results to. Default is to 
            create a new dataframe.
    
        Returns
        -------
        DataFrame: The evaluate_df with the results appended.
    '''

    if evaluate_df is None:
        evaluate_df = pd.DataFrame()

    df = pd.DataFrame(results, index = [name])
    
    return evaluate_df.append(df)

################################################################################

def evaluate(target: pd.Series, prediction: pd.Series, positive_label: str, prefix: str = ''):
    return {
        prefix + 'accuracy' : round(accuracy_score(target, prediction), 2),
        prefix + 'roc_auc' : round(roc_auc_score(target, prediction), 2)
    }