import logging

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

from src.linear_regression_gd import LinearRegressionGD


class LinearRegressionSGD(LinearRegressionGD):
    """Linear Regression using Stochastic Gradient Descent"""

    def __init__(
        self,
        learning_rate=0.01,
        num_epochs=50,
        tolerance=1e-6,
        verbose=True,
        rng_seed: int | None = None,
    ):
        super().__init__(learning_rate, num_epochs, tolerance, verbose)
        self.num_epochs = num_epochs
        self._rng = np.random.default_rng(rng_seed)

    def fit(self, X, y):
        m, d = X.shape
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        self.theta = np.zeros(d + 1)

        self.loss_history = []
        self.loss_history.append(
            self._compute_loss(X_with_intercept, y, self.theta)
        )
        for epoch in tqdm(range(self.num_epochs)):
            indices = self._rng.permutation(m)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]

            for i in range(m):
                xi = X_shuffled[i].reshape(1, -1)
                yi = y_shuffled[i]
                prediction = xi @ self.theta
                self._current_gradient = xi.T @ (prediction - yi)

                self._update_theta()

            epoch_loss = self._compute_loss(X_with_intercept, y, self.theta)
            self.loss_history.append(epoch_loss)

            if self.verbose and (epoch % 100 == 0):
                logger.info(
                    "Epoch %s, Loss: %.6f, Theta: %s",
                    epoch,
                    epoch_loss,
                    self.theta,
                )

            if self._is_early_stopped():
                break

        return self
