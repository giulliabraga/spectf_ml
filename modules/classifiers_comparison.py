import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline
import statistics as st

class ClassifiersComparison():
    """
    Implements several methods to perform the comparison of multiple classifiers.

    Parameters:
    X (array-like): Dataset features.
    y (array-like): Dataset target variable.
    model: The classifier model to be used.
    param_dict (dict, optional): Dictionary of hyperparameters for fine-tuning. Defaults to None.

    Methods:
    - 
    -

    """

    def __init__(self, X, y, model, param_dict=None):
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
                                        cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=True),
                                        scoring='accuracy', 
                                        refit = True,
                                        n_jobs=-1, 
                                        random_state=0,
                                        verbose=1 
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

        features = self.X.values
        target = self.y.values

        # Lists to store the metrics
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

        for rnd_state in range(30):

            print(f'Iteration {rnd_state}')

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rnd_state)

            for fold, (train_idx, test_idx) in enumerate(skf.split(features, target)):
            
                print(f'Fold: {fold}')

                # Train and test split for this particular fold
                X_train_split, y_train_split = features[train_idx], target[train_idx]
                X_test_split, y_test_split = features[test_idx], target[test_idx]

                # Fitting model

                if finetune: #You can choose to fine-tune it every fold

                    # Maybe the user did not define the parameters dictionary. In this case you cannot perform the search.
                    if self.param_dict == None:
                        raise Exception("The 'params_dict' parameter was not defined.")
                    
                    # If it was defined correctly, the fine-tuning will be performed
                    else:
                        print('Hyperparameter fine-tuning...\n')
                        fitted_model = self.fine_tuning_pipeline()

                        rnd_cv_results = fitted_model.fit(X_train_split, y_train_split)

                        best_params = rnd_cv_results.named_steps['randomizedsearchcv'].best_params_
                        best_score = rnd_cv_results.named_steps['randomizedsearchcv'].best_score_

                        print(f'Best parameters: {best_params} \nBest accuracy: {best_score}')

                else: # Or just fit it without any fine-tuning
                    print('Fitting without fine-tuning...\n')
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
                
        metrics = pd.DataFrame(metrics_per_split)

        print(f'\n Metrics: \n{metrics}')
        
        return metrics
    
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