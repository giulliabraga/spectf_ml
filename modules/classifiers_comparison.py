import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import statistics as st

class ClassifiersComparison():
    """
    Implements several methods to perform the comparison of multiple classifiers.

    Parameters:
    dataset (DataFrame): Dataset containing features and target variable.

    Methods:
    - calculate_metrics:
    """

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.classifiers = []
        self.best_params = []
        self.model = model
        

    def metrics_estimation(self, metric):
        """
        Get an estimate and confidence interval for a classifier evaluation metric.

        Parameters:
        metric (list): All observations for a certain metric.

        Returns:
        estimate (float): The point estimate for an evaluation metric.
        confidence_interval (string): The 95% confidence interval for this metric.

        """


        estimate = {}
        return estimate

    def make_pipeline(self):
        """
        Creation of a preprocessing and classifier pipeline.

        Returns:
        Pipeline: A scikit-learn pipeline containing a scaler and a classifier.
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', self.model_create)
        ])

    def cross_validation(self, X, y, pipeline):
        """
        To perform a 30 x 10-fold cross-validation, the following strategy was adopted: 
        - For each 10-fold cross-validation, the scikit function StratifiedKFold was used. 
        - To get 30 reproductions, random states from 0 to 29 were used.
    
        Parameters:
        """

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        # Lists to store the metrics obtained per split
        metrics_per_split = {
            'train_accuracy': [],
            'test_accuracy': [],
            'f1_score': [],
            'AUC': [],
            'precision': [],
            'recall': [],        
        }

        fold = 0
        for train_idx, test_idx in cv.split(X, y):
        
            print(f'Fold: {fold}')

            # Train and test split for this particular fold
            X_train_split, y_train_split = X[train_idx], y[train_idx]
            X_test_split, y_test_split = X[test_idx], y[test_idx]

            # Fitting model
            fitted_model = pipeline.fit(X_train_split,y_train_split)

            # Train and test predictions
            y_train_pred_split = fitted_model.predict(X_train_split)
            y_test_pred_split = fitted_model.predict(X_test_split)

            # Calculate metrics
            train_accuracy = accuracy_score(y_train_split, y_train_pred_split)
            test_accuracy = accuracy_score(y_test_split, y_test_pred_split)
            f1 = f1_score(y_test_split, y_test_pred_split)
            auc = roc_auc_score(y_test_split, y_test_pred_split)
            prec = precision_score(y_test_split, y_test_pred_split)
            rec = recall_score(y_test_split, y_test_pred_split)

            # Storing metrics
            metrics_per_split['train_accuracy'].append(train_accuracy)
            metrics_per_split['test_accuracy'].append(test_accuracy)
            metrics_per_split['f1_score'].append(f1)
            metrics_per_split['AUC'].append(auc)
            metrics_per_split['precision'].append(prec)
            metrics_per_split['recall'].append(rec)

            fold = fold + 1

        metrics_df = pd.DataFrame(metrics_per_split)

        return metrics_df


    def hyperparameter_tuning(self, X, y, pipeline, params):
        """
        This method implements a 5-fold hyperparameter optimization in the remaining 9 folds for each step in the 30 x 10-fold cross-validation.

        Parameters:
        X (array-like): Training features.
        y (array-like): Training labels.
        pipeline (Pipeline): A scikit-learn pipeline.

        Returns:
        best_params (dict): The best hyperparameters for the classifier.
        """

        rnd_search = RandomizedSearchCV(
            pipeline, 
            params, 
            cv=StratifiedKFold(n_splits=5, random_state=0),
            scoring='accuracy', 
            n_jobs=-1, 
            random_state=0, # For better reproductibility
            verbose=0 
            )
        
        rnd_search.fit(X, y)

        # The best parameters will be used to initialize a new model and retrain it with the remaining 9-folds
        best_params = rnd_search.best_params_ 

        return best_params
    
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


if __name__ == "__main__":
    dataset = pd.read_csv('../data/preprocessed/SPECTF_preprocessed.csv')

    comparison = ClassifiersComparison(dataset)