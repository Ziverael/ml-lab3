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
        tolerance=1e-6,
        verbose=True,
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.theta = None
        self.loss_history = []

    def _compute_loss(self, X, y, theta):
        """Compute MSE loss"""
        m = len(y)
        predictions = X @ theta
        return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    def _compute_gradient(self, X, y, theta):
        m = len(y)
        predictions = X @ theta
        gradient = (1 / m) * X.T @ (predictions - y)
        return gradient

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
            gradient = self._compute_gradient(X_with_intercept, y, self.theta)
            self._update_theta(gradient)
            if not (idx % 100):
                if self.verbose:
                    logger.info("Theta: %.02f", self.theta)
                else:
                    logger.debug("Theta: %.02f", self.theta)
            self.loss_history.append(
                self._compute_loss(X_with_intercept, y, self.theta)
            )
            if idx > 0:
                # if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance: This is too slow for tests
                if  np.linalg.norm(gradient) < self.tolerance:
                    print(self.loss_history)
                    if self.verbose:
                        logger.info("Converged at iteration: %s.", idx)
                    break

    def _update_theta(self, gradient) -> None:
        self.theta = self.theta - self.learning_rate * gradient

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


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    m = 100  # number of samples
    X = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(m)  # y = 4 + 3x + noise

    # Fit using gradient descent
    model_gd = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=True
    )
    model_gd.fit(X, y)

    # Compare with closed-form solution
    X_with_intercept = np.column_stack([np.ones(m), X])
    theta_closed_form = (
        np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        @ X_with_intercept.T
        @ y
    )

    print("\nGradient Descent Solution:")
    print(f"θ₀ (intercept) = {model_gd.theta[0]:.4f}")
    print(f"θ₁ (slope) = {model_gd.theta[1]:.4f}")

    print("\nClosed-Form Solution:")
    print(f"θ₀ (intercept) = {theta_closed_form[0]:.4f}")
    print(f"θ₁ (slope) = {theta_closed_form[1]:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Plot data and fitted line
    ax1.scatter(X, y, alpha=0.5, label="Data")
    X_plot = np.array([[0], [2]])
    y_pred_gd = model_gd.predict(X_plot)
    ax1.plot(X_plot, y_pred_gd, "r-", linewidth=2, label="Gradient Descent")
    ax1.plot(
        X_plot,
        [theta_closed_form[0], theta_closed_form[0] + 2 * theta_closed_form[1]],
        "g--",
        linewidth=2,
        label="Closed-Form",
    )
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Linear Regression Fit", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss history
    ax2.plot(model_gd.loss_history, linewidth=2)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Loss J(θ)", fontsize=12)
    ax2.set_title("Loss Function Convergence", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Generate synthetic data with multiple features
    np.random.seed(42)
    m = 1000  # samples
    d = 5     # features

    X = np.random.randn(m, d)
    true_theta = np.array([2.0, 1.5, -3.0, 0.5, 2.5, -1.0])  # [intercept, features]
    y = true_theta[0] + X @ true_theta[1:] + 0.5 * np.random.randn(m)

    # Fit using gradient descent
    model_gd = LinearRegressionGD(learning_rate=0.01, num_iterations=2000, verbose=False)
    model_gd.fit(X, y)

    # Closed-form solution
    X_with_intercept = np.column_stack([np.ones(m), X])
    theta_closed_form = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    # Compare results
    print("Parameter Comparison:")
    print(f"{'Parameter':<12} {'True Value':<12} {'Gradient Descent':<18} {'Closed-Form':<12}")
    print("-" * 60)
    print(f"{'θ₀':<12} {true_theta[0]:<12.4f} {model_gd.theta[0]:<18.4f} {theta_closed_form[0]:<12.4f}")
    for i in range(1, d+1):
        print(f"{'θ'+str(i):<12} {true_theta[i]:<12.4f} {model_gd.theta[i]:<18.4f} {theta_closed_form[i]:<12.4f}")

    # Plot convergence
    model_gd.plot_loss_history()