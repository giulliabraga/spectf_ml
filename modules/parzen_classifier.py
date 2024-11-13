from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class ParzenClassifier(ClassifierMixin, BaseEstimator):
    """
    Parzen-based classifier using Gaussian kernel density estimation for classification problems.

    Parameters
    bandwidth (float): The smoothing parameter or window width for the Parzen window.
    n_class (int): The number of unique classes in the dataset.
    **params (dict): Additional parameters.

    Attributes
    means (np.array): List containing the mean of features for each class.
    x_class (list of np.array): List of feature arrays, one for each class, containing training examples.
    bandwidth (float): The window width for the Parzen estimator.
    """

    def __init__(self, bandwidth=1.0, n_class=2, **params):
        super().__init__()
        self.n_class = n_class
        self.means = []
        self.x_class = []
        self.bandwidth = bandwidth 

    def fit(self, X: np.array, y: np.array):
        """
        Selects the data corresponding to each class and stores it.

        Parameters
        X (np.array): The training data with shape (n_samples, n_features)
        y (np.array):  Class labels for each sample in X

        Returns
        self (object): Fitted estimator
        """
        for c in range(self.n_class):
            _X = X[y == c]
            if _X.shape[0] > 0:
                self.means.append(_X.mean(0))
                self.x_class.append(_X)
        return self

    def prod_class(self, c, h):
        """
        This method is the core of the classifier, computing the probability density estimate for each class according to the stipulated equation.

        Parameters
        c (int): Class index.
        h (float): Bandwidth for the Gaussian kernel.

        Returns
        prod (function): A function that takes data X and returns Parzen densities for class c.
        """
        def prod(X):
            n, p = self.x_class[c].shape  # Number of samples (n) and features (p)
            dif = np.abs(X - self.x_class[c]) / h  # Compute the differences scaled by h
            dif_gaussian = np.exp(-0.5 * (dif ** 2)) / ((2 * np.pi) ** (1 / 2))  # Gaussian kernel formula
            dif_parzen = np.prod(dif_gaussian, 1)  # Product over all features
            return np.sum(dif_parzen) / (n * (h ** p))  # Normalize by N and h^p
        
        return prod

    def prod_windows_parzen(self, X):
        """
        Computes the Parzen density for all classes over input data X.

        Parameters
        X (np.array): The input data.

        Returns
        parzen_estimates (np.array): Density scores for each class and each sample.
        """

        h = self.bandwidth

        parzen_estimates = np.array([np.apply_along_axis(self.prod_class(i, h), axis=1, arr=X) for i in range(self.n_class)]).T
        
        return parzen_estimates

    def predict(self, X):
        """
        Predicts the class labels for the provided data.

        Parameters
        X (np.array): Input data.

        Returns
        predicted_labels (np.array): Predicted class labels for each sample.
        """

        predicted_labels = self.prod_windows_parzen(X).argmax(1)
        
        return predicted_labels