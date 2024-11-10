import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

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

    def __init__(self, list_of_model_results, list_of_model_names, list_of_metrics_names):
        self.list_of_model_results = list_of_model_results
        self.list_of_model_names = list_of_model_names
        self.list_of_metrics_names = list_of_metrics_names

    def get_estimate_and_ci(self, parameter):
        """
        Get a point estimate and a confidence interval for an array of observations.

        Parameters:
        parameter (array-like): Array containing observations for a certain parameter.

        Outputs:
        mean_estimate (float): The mean point estimate for this parameter.
        std_deviation (float): The sample's standard deviation.
        confidence_interval (string): The confidence interval for this parameter.
        """
        # Point estimate for the parameters mean
        mean_estimate = np.mean(parameter)

        # Sample std
        std_deviation = np.std(parameter)

        # 95% confidence interval
        E = 1.96 * (std_deviation / len(parameter))
        confidence_interval = f'{mean_estimate - E}, mean, {mean_estimate + E}'

        return mean_estimate, std_deviation, confidence_interval
    
    def concat_dataframes(self, list_of_dfs, list_of_ids):
        '''
        With a list of DataFrames and a list of identifiers to each of them, concatenate them while identifying each one.

        Parameters:

        list_of_dfs (list of DataFrames): List of DataFrames to be concatenated. They must have the same column names.
        list_of_ids (list): List of identifiers, which could be strings or numbers.

        Outputs:
        concat_df (DataFrame): DataFrame containing a set of concatenated dfs, identified by the column 'ID'.
        '''

        concat_df = pd.DataFrame()

        for k, df in enumerate(list_of_dfs):
            df['ID'] = list_of_ids[k]
            concat_df = pd.concat([concat_df, df], ignore_index=True)
        
        return concat_df

    def friedman_test(self):
        '''
        Use the Friedman test (non-parametric) to compare the classifiers.
        '''

        list_of_statistics = []
        list_of_pvalues = []
        list_of_diff_flags = []

        concat_df = self.concat_dataframes(self.list_of_model_results, self.list_of_model_names)

        for metric in self.list_of_metrics_names:

            metric_by_model = [concat_df[metric][concat_df['ID'] == id_value].tolist() for id_value in self.list_of_model_names]

            statistic, p_value = friedmanchisquare(*metric_by_model)

            if p_value < 0.05:
                are_different = True
            else:
                are_different = False

            '''
            print(f'Metric: {metric}')
            print(f'Statistic: {statistic}')
            print(f'p-value: {p_value}')
            print(f'It is {are_different} that the classifiers performance for this metric is signifficantly different.')
            '''

            list_of_statistics.append(statistic)
            list_of_pvalues.append(p_value)
            list_of_diff_flags.append(are_different)


        friedman_dict = {
            'metric': self.list_of_metrics_names,
            'statistic': list_of_statistics,
            'p-value': list_of_pvalues,
            'are_different': list_of_diff_flags
        }

        friedman_results = pd.DataFrame(friedman_dict)

        metrics_with_difference = friedman_results[friedman_results['are_different'] == True]['metric'].tolist()

        return friedman_results, metrics_with_difference
    
    def nemenyi_test(self, metrics_with_difference):
        '''
        Use the Nemenyi post-hoc test to compare the classifiers.
        '''

        list_of_metrics = []
        list_of_pvalues = []
        list_of_classifier_pairs = []
        list_of_diff_flags = []

        concat_df = self.concat_dataframes(self.list_of_model_results, self.list_of_model_names)

        for metric in metrics_with_difference:

            metric_by_model = [concat_df[metric][concat_df['ID'] == id_value].tolist() for id_value in self.list_of_model_names]

            nemenyi_groups = np.array(metric_by_model).T

            nemenyi_output = posthoc_nemenyi_friedman(nemenyi_groups)

            for i in range(len(nemenyi_output)):
                for j in range(i + 1, len(nemenyi_output)):

                    p_value = nemenyi_output.iloc[i, j]

                    classifier1_id = nemenyi_output.index[i]
                    classifier1_name = self.list_of_model_names[classifier1_id]

                    classifier2_id = nemenyi_output.columns[j]
                    classifier2_name = self.list_of_model_names[classifier2_id]

                    classifier_pair = f'{classifier1_name} vs {classifier2_name}'

                    if p_value < 0.05:
                        are_different = True
                    else: 
                        are_different = False

                    # Adding this row to the corresponding lists
                    list_of_metrics.append(metric)
                    list_of_classifier_pairs.append(classifier_pair)
                    list_of_pvalues.append(p_value)
                    list_of_diff_flags.append(are_different)

        nemenyi_dict = {
            'metric': list_of_metrics,
            'classifier_pair': list_of_classifier_pairs,
            'p-value': list_of_pvalues,
            'are_different': list_of_diff_flags
        }

        nemenyi_results = pd.DataFrame(nemenyi_dict)

        return nemenyi_results