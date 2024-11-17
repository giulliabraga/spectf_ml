import numpy as np
from sklearn.neighbors import KernelDensity


class KernelDensityClassifier:
    """
    Kernel Density Classifier for binary classification.

    Parameters
    ----------
    kernel : str, optional (default='gaussian')
        The kernel to use for the Kernel Density Estimation.
    bandwidth : float, optional (default=0.5)
        The bandwidth of the kernel.

    Attributes
    ----------
    kde_positive : KernelDensity
        Kernel Density Estimator for the positive class.
    kde_negative : KernelDensity
        Kernel Density Estimator for the negative class.
    """

    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.kde_positive = None
        self.kde_negative = None

    def fit(self, X, y):
        """
        Fit the Kernel Density Estimators for the positive and negative classes.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values (1 for positive class, 0 for negative class).
        """
        X_positive = X[y == 1]
        X_negative = X[y == 0]

        self.kde_positive = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kde_negative = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)

        self.kde_positive.fit(X_positive)
        self.kde_negative.fit(X_negative)

    def predict_proba(self, X_new):
        """
        Estimate the probabilities of the positive class for the new data.

        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features)
            New data.

        Returns
        -------
        probabilities : array, shape (n_samples,)
            Estimated probabilities of the positive class.
        """
        if self.kde_positive is None or self.kde_negative is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with appropriate data.")
        
        log_density_positive = self.kde_positive.score_samples(X_new)
        log_density_negative = self.kde_negative.score_samples(X_new)
        
        density_positive = log_density_positive
        density_negative = log_density_negative
        
        return np.vstack((density_negative, density_positive)).T

    def predict(self, X_new):
        """
        Predict the class labels for the new data.

        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features)
            New data.

        Returns
        -------
        predictions : array, shape (n_samples,)
            Predicted class labels (1 for positive class, 0 for negative class).
        """
        densities = self.predict_proba(X_new)
        return np.argmax(densities, axis=1)