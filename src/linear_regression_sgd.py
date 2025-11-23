import numpy as np
from tqdm import tqdm
import logging

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
    ):
        super().__init__(learning_rate, num_epochs, tolerance, verbose)
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

        # Initial loss
        self.loss_history = []
        self.loss_history.append(self._compute_loss(X_with_intercept, y, self.theta))

        for epoch in tqdm(range(self.num_epochs)):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]

            # SGD updates one example at a time
            for i in range(m):
                xi = X_shuffled[i].reshape(1, -1)
                yi = y_shuffled[i]
                prediction = xi @ self.theta
                gradient = xi.T @ (prediction - yi)

                self._update_theta(gradient)

            # Compute full loss for monitoring
            epoch_loss = self._compute_loss(X_with_intercept, y, self.theta)
            self.loss_history.append(epoch_loss)

            if self.verbose and (epoch % 5 == 0):
                logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.6f}, Theta: {self.theta}")

            # Early stopping
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                    break

        return self
