import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearRegressionGD:
    """Linear Regression using Batch Gradient Descent"""

    def __init__(
        self,
        learning_rate=0.01,
        num_iterations=1000,
        tolerance=1e-10,
        verbose=True,
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.theta = None
        self.loss_history = []
        self._current_gradient: np.ndarray = np.zeros((1, 1))

    def _compute_loss(self, X, y, theta):
        """Compute MSE loss"""
        m = len(y)
        predictions = X @ theta
        return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    def _compute_gradient(self, X, y, theta):
        m = len(y)
        predictions = X @ theta
        return (1 / m) * X.T @ (predictions - y)

    def fit(self, X, y):
        _, d = X.shape
        X_with_intercept = (
            np.ones((X.shape[0], 1))
            if X.shape[1] == 0
            else np.column_stack([np.ones(X.shape[0]), X])
        )
        self.theta = np.zeros(d + 1, dtype=np.float64)
        loss = self._compute_loss(X_with_intercept, y, self.theta)
        self.loss_history.append(loss)
        for idx in tqdm(range(self.num_iterations)):
            self._current_gradient = self._compute_gradient(
                X_with_intercept, y, self.theta
            )
            self._update_theta()
            if not (idx % 100):
                self._log_progress()
            self.loss_history.append(
                self._compute_loss(X_with_intercept, y, self.theta)
            )
            if self._is_early_stopped():
                break

    def _log_progress(self) -> None:
        if self.verbose:
            logger.info("Theta: %s", ", ".join(f"{v:.2f}" for v in self.theta))
        else:
            logger.debug("Theta: %s", ", ".join(f"{v:.2f}" for v in self.theta))

    def _update_theta(self) -> None:
        self.theta = self.theta - self.learning_rate * self._current_gradient

    def _is_early_stopped(self) -> bool:
        if len(self.loss_history) > 0:
            if (
                abs(self.loss_history[-1] - self.loss_history[-2])
                < self.tolerance
            ):  # This is too slow for tests
                # if np.linalg.norm(self._current_gradient) < self.tolerance:
                print(self.loss_history)
                if self.verbose:
                    logger.info(
                        "Converged at iteration: %s.", len(self.loss_history)
                    )
                return True
        return False

    def predict(self, X):
        X_with_intercept = (
            np.ones((X.shape[0], 1))
            if X.shape[1] == 0
            else np.column_stack([np.ones(X.shape[0]), X])
        )
        return X_with_intercept @ self.theta

    def plot_loss_history(self):
        """Plot the loss function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Loss J(θ)", fontsize=12)
        plt.title("Loss Function vs. Iteration", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()
