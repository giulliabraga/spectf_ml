import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline
import statistics as st

class StatisticalMethods():
    """
    Implements several methods to perform the statistical comparison of multiple classifiers.

    Parameters:
    X (array-like): Dataset features.
    y (array-like): Dataset target variable.
    model: The classifier model to be used.
    param_dict (dict, optional): Dictionary of hyperparameters for fine-tuning. Defaults to None.

    Methods:
    - get_estimate_and_ci
    - friedman_test
    - nemenyi_test

    """

    def __init__(self, metric):
        self.metric = metric

    def get_estimate_and_ci(self, parameter):
        """
        Get a point estimate and a confidence interval for an array of observations.

        Parameters:
        parameter (array): Array containing observations for a certain parameter.

        Outputs:
        point_estimate (float): The point estimate for this parameter.
        confidence_interval (string): The confidence interval for this parameter.
        """

        point_estimate = []
        confidence_interval = []

        return point_estimate, confidence_interval

    def friedman_test(self, metrics):
        '''
        Use the Friedman test (non-parametric) to compare the classifiers.
        '''

        friedman_results = {}

        return friedman_results
    
    def nemenyi_test(self, metrics):
        '''
        Use the Nemenyi post-hoc test to compare the classifiers.
        '''

        nemenyi_results = {}

        return nemenyi_results