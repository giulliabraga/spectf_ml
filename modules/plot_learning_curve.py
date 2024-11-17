import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score

class LearningCurvePlotter:
    """
    A class to plot learning curves for a QDA classifier using different evaluation metrics,
    and store the results in a DataFrame.

    Attributes:
        classifier (sklearn.naive_bayes.QuadraticDiscriminantAnalysis): The QDA classifier.
        X (array-like): The feature matrix.
        y (array-like): The target vector.
        metrics (list): List of evaluation metrics to plot.
        results_df (pd.DataFrame): A DataFrame storing the evaluation metrics for each training size.
    """

    def __init__(self, X, y, metrics=None):
        """
        Initializes the LearningCurvePlotter with the data and selected metrics.

        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
            metrics (list, optional): The evaluation metrics to use. Default is [test_accuracy, error_rate, coverage, f1_score].
        """
        self.X = X
        self.y = y
        self.classifier = QuadraticDiscriminantAnalysis()
        self.metrics = metrics if metrics else ['test_accuracy', 'error_rate', 'coverage', 'f1_score']
        self.results_df = pd.DataFrame()

    def plot_learning_curve(self):
        """
        Plots learning curves for each evaluation metric, training the classifier with varying training sizes.
        Also stores the results in the DataFrame.
        """
        # Split the data into 70% training and 30% testing
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.X, self.y, test_size=0.30, stratify=self.y
        )

        train_sizes = np.arange(0.05, 1.00, 0.05)  # Training sizes from 5% to 100%
        results = {metric: [] for metric in self.metrics}
        table_data = []

        for train_size in train_sizes:
            # Subsample the training set
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full, train_size=train_size, stratify=y_train_full
            )
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)

            row = {'train_size': f"{train_size*100:.1f}%"}

            for metric in self.metrics:
                if metric == 'test_accuracy':
                    value = accuracy_score(y_test, y_pred)
                    results['test_accuracy'].append(value)
                    row['test_accuracy'] = value
                elif metric == 'error_rate':
                    value = 1 - accuracy_score(y_test, y_pred)
                    results['error_rate'].append(value)
                    row['error_rate'] = value
                elif metric == 'coverage':
                    coverage = np.mean(np.in1d(y_test, y_pred))
                    results['coverage'].append(coverage)
                    row['coverage'] = coverage
                elif metric == 'f1_score':
                    value = f1_score(y_test, y_pred, average='weighted')
                    results['f1_score'].append(value)
                    row['f1_score'] = value

            table_data.append(row)

        self.results_df = pd.DataFrame(table_data)  # Store results in DataFrame
        self._plot(results, train_sizes)

    def _plot(self, results, train_sizes):
        """
        Helper method to plot the results of the learning curve.

        Args:
            results (dict): The dictionary containing the evaluation results.
            train_sizes (array-like): The array of training sizes.
        """
        plt.figure(figsize=(10, 6))
        for metric, scores in results.items():
            plt.plot(train_sizes * 100, scores, label=metric,marker='.')

        plt.title('Learning Curves for QDA')
        plt.xlabel('Training Size (%)')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_results(self):
        """
        Returns the DataFrame containing the evaluation metrics for each training size.
        
        Returns:
            pd.DataFrame: A DataFrame with the metrics for each training size.
        """
        return self.results_df
