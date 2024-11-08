import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline
import statistics as st

class ClassifiersComparison():
    """
    Implements several methods to perform the comparison of multiple classifiers.

    Parameters:
    X (array-like): Dataset features.
    y (array-like): Dataset target variable

    Methods:
    - 
    -

    """

    def __init__(self, X, y, model, param_dict):
        self.X = X
        self.y = y
        self.model = model
        self.param_dict = param_dict

    def fine_tuning_pipeline(self):
        """
        Creation of a preprocessing and classifier pipeline for fine-tuning.

        Outputs:
        Pipeline: A scikit-learn pipeline containing a scaler and a randomized search cross-validation class for a classifier's hyperparameter fine-tuning.
        """

        optimized_model = make_pipeline(StandardScaler(), 
                                        RandomizedSearchCV(
                                        self.model, 
                                        self.param_dict, 
                                        cv=StratifiedKFold(n_splits=5, random_state=0),
                                        scoring='accuracy', 
                                        refit = True,
                                        n_jobs=-1, 
                                        random_state=0,
                                        verbose=0 
                                        )
                                    )

        return optimized_model

    def cross_validation(self, n_folds=10, finetune = True):
        """
        To perform a 30 x 10-fold cross-validation, the following strategy was adopted: 
        - For each 10-fold cross-validation, the scikit function StratifiedKFold was used. 
        - To get 30 reproductions, random states from 0 to 29 were used.
    
        Parameters:
        """

        features = self.X
        labels = self.y

        for rnd_state in range(30):

            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rnd_state)

            # Lists to store the metrics obtained per split
            metrics_per_split = {
                'iteration': [],
                'fold': [],
                'train_accuracy': [],
                'test_accuracy': [],
                'f1_score': [],
                'AUC': [],
                'precision': [],
                'recall': [],        
            }

            fold = 0

            for train_idx, test_idx in cv.split(features, labels):
            
                print(f'Fold: {fold}')

                # Train and test split for this particular fold
                X_train_split, y_train_split = features[train_idx], labels[train_idx]
                X_test_split, y_test_split = features[test_idx], labels[test_idx]

                # Fitting model
                if finetune:
                    fitted_model = self.hyperparameter_tuning(self.model)
                    fitted_model.fit(X_train_split, y_train_split)
                else: 
                    fitted_model = self.model
                    fitted_model.fit(X_train_split, y_train_split)

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
                metrics_per_split['fold'].append(f'fold_{fold}')
                metrics_per_split['iteration'].append(f'iteration_{rnd_state}')

                fold = fold + 1

        metrics_df = pd.DataFrame(metrics_per_split)
        
        return metrics_df
    
    def metrics_estimation(self, metric):
        """
        Get an estimate and confidence interval for a classifier evaluation metric.

        Parameters:
        metric (list): All observations for a certain metric.

        Outputs:
        estimate (float): The point estimate for an evaluation metric.
        confidence_interval (string): The 95% confidence interval for this metric.

        """


        estimate = {}
        return estimate
    
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