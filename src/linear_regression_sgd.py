import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
from src.linear_regression_gd import LinearRegressionGD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearRegressionSGD(LinearRegressionGD):
    """Linear Regression using Batch Stochastic Gradient Descent"""

    def fit(self, X, y):
        _, d = X.shape
        X_with_intercept = (
            np.ones((X.shape[0], 1))
            if X.shape[1] == 0
            else np.column_stack([np.ones(X.shape[0]), X])
        )
        self.theta = np.zeros(d + 1, dtype=np.float64)
        self.loss_history.append(
            self._compute_loss(X_with_intercept, y, self.theta)
        )
        for idx in tqdm(range(self.num_iterations)):
            np.random.shuffle(X_with_intercept)
            test_example = X_with_intercept[0, :]
            gradient = self._compute_gradient(test_example, y, self.theta)
            self._update_theta(gradient)
            if not (idx % 100):
                if self.verbose:
                    logger.info("Theta: %.02f", self.theta)
                else:
                    logger.debug("Theta: %.02f", self.theta)
                self.loss_history.append(
                    self._compute_loss(test_example, y, self.theta)
                )
                if (
                    np.abs(sum(gradient)) < self.tolerance
                    or np.abs(self.loss_history[-1] - self.loss_history[-2])
                    < self.tolerance
                ):
                    break


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    m = 100  # number of samples
    X = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(m)  # y = 4 + 3x + noise

    # Fit using gradient descent
    model_gd = LinearRegressionSGD(
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
