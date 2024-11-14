from sklearn.preprocessing import MinMaxScaler
import numpy as np
from time import time
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

class KFCM_K_W_1:
    def __init__(self, c, m=1.1, epochs=100, tol=1e-6, seed=0):
        # Hyperparameters
        self.m = m
        self.epochs = epochs
        self.tol = tol
        self.c = c # c: number of clusters
        self._zero = 1e-20
        self._rng = np.random.default_rng(seed)
        self._epoch = 0
        self._x = None
        self._y = None
        self._n = None
        self._p = None
        self._s = None
        self._u = None
        self._u_m = None
        self._u_m_kernel = None
        self._kernel = None
        self._denominator_j = None
        self._g = None
        self._j_new = None
        self._j_old = None
        return

    def fit(self, X, y=None):
        start_time = time()
        # Normalization
        scaler = MinMaxScaler()
        self._x = scaler.fit_transform(X)[:, :, np.newaxis]
        self._y = np.asarray(y)[:, np.newaxis]

        # n: number of instances
        # p: number of features
        self._n, self._p, _ = self._x.shape

        # Arrays initializations
        # s: array of width parameters 1/s^2, from Step 7
        self._s = np.ones((1,self._p))
        self._u = np.zeros((self._n, self.c, 1))

        # Prototype selection, Step 8
        g_idx = self._rng.integers(0, self._n, self.c)
        self._g = self._x[g_idx]

        self._update_kernel()

        # Compute the membership degree, Step 9,21
        self._update_u()

		# Compute the objective function, Step 11,23
        self._update_j()

        for epoch in range(1, self.epochs + 1):
            self._epoch = epoch
            # Step 13
            self._j_old = self._j_new

            # Step 15
            self._update_s()

            # Step 18
            self._update_g()

            # Step 21
            self._update_u()

            # Step 23
            #if the absolute difference between self._j_new and self._j_old is
            # less than self.tol, the loop will break
            self._update_j()
            if abs(self._j_new - self._j_old) < self.tol:
                break
        end_time = time()
        print(f"Execution time: {round((end_time - start_time) / 60, 2)} minutes")

        return


    def evaluate(self, metric):
        metrics = {
            "accuracy": self._evaluate_accuracy,
            "MPC": self._evaluate_modified_partition_coefficient,
            "rand": self._evaluate_adjusted_rand_score,
            "error": self._evaluate_error,
        }
        metric_function = metrics.get(metric, "error")
        return metric_function()

    def _update_kernel(self):
        """Equation 10."""
        # Step 1: Calculate squared Euclidean distances
        squared_distances = (self._x - self._g.T) ** 2

        # Step 2: Apply weights
        weighted_distances = squared_distances * self._s.T

        # Step 3: Sum across dimensions
        summed_distances = weighted_distances.sum(axis=1)

        # Step 4: Apply the exponential function
        exponential = np.exp(summed_distances * (-1 / 2))
        self._kernel = exponential[:, :, np.newaxis]
        return

    def _update_u(self):
        """Equation 16a."""

        self._denominator_j = 2 - 2 * self._kernel
        self._denominator_j = np.maximum(self._denominator_j, self._zero)

        numerator = np.swapaxes(self._denominator_j, 1, 2)

        division = (numerator / self._denominator_j) ** (1 / (self.m - 1))

        self._u = division.sum(axis=1) ** -1
        self._u = self._u[:, :, np.newaxis]

        self._u_m = self._u**self.m
        self._u_m_kernel = self._u_m * self._kernel
        return

    def _update_j(self):
        """Equation 11."""
        self._j_new = (self._u_m * self._denominator_j).sum()
        print(f"Epoch: {self._epoch:03d} | Objective function J: {self._j_new:.8f}")
        return

    def _update_s(self):
        """Equation 14a."""
        squared_distances = (self._x - self._g.T) ** 2

        squared_distances = np.swapaxes(squared_distances, 1, 2)

        denominator = (self._u_m_kernel * squared_distances).sum(axis=0)[:, :, np.newaxis]
        
        denominator = (denominator).sum(axis=2)[:, :, np.newaxis] # Linha nova

        denominator = np.log(denominator)

        numerator = denominator.sum(axis=1)[:, :, np.newaxis] * self._p**-1

        subtracted = numerator - denominator

        self._s = np.exp(subtracted)

        self._update_kernel()
        return

    def _update_g(self):
        """Equation 15a."""
        x = np.swapaxes(self._x, 1, 2)

        numerator = (self._u_m_kernel * x).sum(axis=0)

        denominator = self._u_m_kernel.sum(axis=0)

        self._g = (numerator / denominator)[:, :, np.newaxis]

        self._update_kernel()
        return

    def _match_clusters(self, y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(conf_matrix, maximize=True)
        mapping = {col: row for row, col in zip(row_ind, col_ind)}
        y_pred_aligned = np.array([mapping[label] for label in y_pred])
        return y_pred_aligned

    def _evaluate_accuracy(self):
        pred = np.argmax(self._u, axis=1).flatten()
        y_true = self._y.flatten()
        pred_aligned = self._match_clusters(y_true, pred)
        accuracy = np.mean(pred_aligned == y_true)
        return accuracy
    
    
    def _evaluate_modified_partition_coefficient(self):
        pc = np.sum(self._u**2) / self._n
        mpc = 1 - (self.c / (self.c - 1)) * (1 - pc)
        return mpc

    def _evaluate_adjusted_rand_score(self):
        y_pred = np.argmax(self._u, axis=1)
        return adjusted_rand_score(self._y[:, 0], y_pred[:, 0])

    def _evaluate_error(self):
        raise ValueError("Metric not implemented")
