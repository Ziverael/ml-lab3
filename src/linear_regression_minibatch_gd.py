import logging

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

from src.linear_regression_gd import LinearRegressionGD


class LinearRegressionMiniBatchGD(LinearRegressionGD):
    """Linear Regression using Mini-Batch Gradient Descent"""

    def __init__(
        self,
        learning_rate=0.01,
        num_epochs=50,
        batch_size=32,
        tolerance=1e-6,
        verbose=True,
    ):
        super().__init__(learning_rate, num_epochs, tolerance, verbose)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y):
        m, d = X.shape
        # Add intercept
        X_with_intercept = (
            np.ones((X.shape[0], 1))
            if X.shape[1] == 0
            else np.column_stack([np.ones(X.shape[0]), X])
        )
        self.theta = np.zeros(d + 1)

        self.loss_history = []
        self.loss_history.append(
            self._compute_loss(X_with_intercept, y, self.theta)
        )

        for epoch in tqdm(range(self.num_epochs)):
            # Shuffle indices
            indices = np.random.permutation(m)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]

            # Mini-batch loop
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute gradient for the batch
                predictions = X_batch @ self.theta
                self._current_gradient = (
                    (1 / len(y_batch)) * X_batch.T @ (predictions - y_batch)
                )

                self._update_theta()

            epoch_loss = self._compute_loss(X_with_intercept, y, self.theta)
            self.loss_history.append(epoch_loss)

            if not (epoch % 5):
                self._log_progress()

            if self._is_early_stopped():
                break

        return self
