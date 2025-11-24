import argparse
from pprint import pprint
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.linear_regression_gd import LinearRegressionGD
from src.linear_regression_minibatch_gd import LinearRegressionMiniBatchGD
from src.linear_regression_sgd import LinearRegressionSGD


def main(type: Literal["gd", "sgd", "minibatch"], args):
    model = {
        "gd": LinearRegressionGD,
        "minibatch": LinearRegressionMiniBatchGD,
        "sgd": LinearRegressionSGD,
    }

    rng = np.random.default_rng(42)
    m = 100  # number of samples
    X = 2 * rng.random((m, 1))

    y = 4 + 3 * X.squeeze() + rng.normal(size=m)  # y = 4 + 3x + noise

    # Fit using gradient descent
    if type == "gd":
        selected_model = model[type](
            learning_rate=0.1, num_iterations=args.num_iterations, verbose=True
        )

    elif type == "sgd":
        selected_model = model[type](
            learning_rate=0.1, num_epochs=args.num_epochs, verbose=True
        )

    elif type == "minibatch":
        selected_model = model[type](
            learning_rate=0.1,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            verbose=True,
        )
    selected_model.fit(X, y)

    # Compare with closed-form solution
    X_with_intercept = np.column_stack([np.ones(m), X])
    theta_closed_form = (
        np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        @ X_with_intercept.T
        @ y
    )

    pprint("\nGradient Descent Solution:")
    pprint(f"θ₀ (intercept) = {selected_model.theta[0]:.4f}")
    pprint(f"θ₁ (slope) = {selected_model.theta[1]:.4f}")

    pprint("\nClosed-Form Solution:")
    pprint(f"θ₀ (intercept) = {theta_closed_form[0]:.4f}")
    pprint(f"θ₁ (slope) = {theta_closed_form[1]:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Plot data and fitted line
    ax1.scatter(X, y, alpha=0.5, label="Data")
    X_plot = np.array([[0], [2]])
    y_pred_gd = selected_model.predict(X_plot)
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
    ax2.plot(selected_model.loss_history, linewidth=2)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Loss J(θ)", fontsize=12)
    ax2.set_title("Loss Function Convergence", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Generate synthetic data with multiple features
    rng = np.random.default_rng(42)
    m = 1000  # samples
    d = 5  # features

    X = rng.normal(size=(m, d))
    true_theta = np.array(
        [2.0, 1.5, -3.0, 0.5, 2.5, -1.0]
    )  # [intercept, features]
    y = true_theta[0] + X @ true_theta[1:] + 0.5 * rng.normal(size=m)

    selected_model.fit(X, y)

    # Closed-form solution
    X_with_intercept = np.column_stack([np.ones(m), X])
    theta_closed_form = (
        np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        @ X_with_intercept.T
        @ y
    )

    # Compare results
    pprint("Parameter Comparison:")
    pprint(
        f"{'Parameter':<12} {'True Value':<12} {'Gradient Descent':<18} {'Closed-Form':<12}"
    )
    pprint("-" * 60)
    pprint(
        f"{'θ₀':<12} {true_theta[0]:<12.4f} {selected_model.theta[0]:<18.4f} {theta_closed_form[0]:<12.4f}"
    )
    for i in range(1, d + 1):
        pprint(
            f"{'θ' + str(i):<12} {true_theta[i]:<12.4f} {selected_model.theta[i]:<18.4f} {theta_closed_form[i]:<12.4f}"
        )

    # Plot convergence
    selected_model.plot_loss_history()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run linear regression training with specified model."
    )
    # required model choice
    parser.add_argument(
        "model",
        type=str,
        help="Model type to use, e.g. 'gd', 'sgd', 'minibatch'.",
    )

    # optional parameters
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="Number of gradient descent iterations.",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (for SGD / minibatch).",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (for minibatch).",
    )

    args = parser.parse_args()
    main(args.model, args)
