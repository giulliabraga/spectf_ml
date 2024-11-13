from sklearn.preprocessing import MinMaxScaler
import numpy as np
from time import time
from sklearn.metrics import adjusted_rand_score
import pandas as pd

class KFCM_K:
    def __init__(self, c, m=1.1, epochs=100, tol=1e-6, seed=0):
        # Hyperparameters
        self.m = m #fuzzy degreen
        self.epochs = epochs
        self.tol = tol
        self.c = c # c: number of clusters
        self._zero = 1e-20
        self._rng = np.random.default_rng(seed)
        self._epoch = 0
        self._x = None #data
        self._y = None
        self._n = None #number of instances
        self._p = None #number of features
        self._u = None #membership degreen of point x to cluster c
        self._u_m = None #u fuzzification
        self._kernel = None #similarity matrix between points
        self._denominator_j = None
        self._g = None #prototype
        self._j_new = None #cost function after update
        self._j_old = None #old cost function
        return

    def fit(self, X, y=None):
        start_time = time()
        # Normalization
        scaler = MinMaxScaler()
        self._x = scaler.fit_transform(X)[:, :, np.newaxis]
        self._y = np.asarray(y)[:, np.newaxis]
        self._n, self._p, _ = self._x.shape
    
        # Arrays initializations
        self._u = np.zeros((self._n, self.c, 1))

        # Prototype selection
        g_idx = self._rng.integers(0, self._n, self.c)
        self._g = self._x[g_idx]

        self._update_kernel()

        # Compute the membership degree
        self._update_u()

        # Compute the objective function
        self._update_j()

        for epoch in range(1, self.epochs + 1):
            self._epoch = epoch
            self._j_old = self._j_new

            # Update prototypes
            self._update_g()

            # Update membership degrees
            self._update_u()

            # Update the objective function
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
        squared_distances = (self._x - self._g.T) ** 2
        summed_distances = squared_distances.sum(axis=1)
        exponential = np.exp(summed_distances * (-1 / 2))
        self._kernel = exponential[:, :, np.newaxis]
        return

    def _update_u(self):
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
        self._j_new = (self._u_m * self._denominator_j).sum()
        print(f"Epoch: {self._epoch:03d} | Objective function J: {self._j_new:.8f}")
        return

    def _update_g(self):
        x = np.swapaxes(self._x, 1, 2)

        numerator = (self._u_m_kernel * x).sum(axis=0)

        denominator = self._u_m_kernel.sum(axis=0)

        self._g = (numerator / denominator)[:, :, np.newaxis]

        self._update_kernel()
        return

    def _evaluate_accuracy(self):
        pred = np.argmax(self._u, axis=1)
        y_with_pred = pd.DataFrame(np.concatenate((self._y, pred), axis=1))
        y_with_pred["value"] = 1
        pivot_table = pd.pivot_table(
            y_with_pred, columns=[0], index=[1], values="value", aggfunc="sum"
        )
        pivot_table = pivot_table.fillna(0).values

        n_i = pivot_table.sum(axis=1)[:, np.newaxis]
        p_ij = pivot_table / n_i
        p_i = p_ij.max(axis=1)[:, np.newaxis]
        acc = (n_i * p_i).sum(axis=0) / self._n
        return acc[0]

    def _evaluate_modified_partition_coefficient(self):
        pc = np.sum(self._u**2) / self._n
        mpc = 1 - (self.c / (self.c - 1)) * (1 - pc)
        return mpc

    def _evaluate_adjusted_rand_score(self):
        y_pred = np.argmax(self._u, axis=1)
        return adjusted_rand_score(self._y[:, 0], y_pred[:, 0])

    def _evaluate_error(self):
        raise ValueError("Metric not implemented")
    
