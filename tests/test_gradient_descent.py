"""
Comprehensive test suite for Gradient Descent Linear Regression implementations.

This module contains unit tests for the LinearRegressionGD, LinearRegressionSGD,
and LinearRegressionMiniBatchGD classes, designed to validate both correctness
and robustness of the implementations.

Test Categories:
1. Basic functionality tests - comparing against closed-form solution
2. Convergence tests - verifying loss decreases over iterations
3. Edge cases - boundary conditions and special cases
4. Mathematical properties - testing known analytical solutions
5. Feature scaling effects - testing convergence with/without scaling
6. Learning rate sensitivity - testing different learning rates
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.linear_regression_gd import LinearRegressionGD
from src.linear_regression_sgd import LinearRegressionSGD
from src.linear_regression_minibatch_gd import LinearRegressionMiniBatchGD


# =============================================================================
# Helper Functions
# =============================================================================


def closed_form_solution(X, y):
    """Compute closed-form solution for comparison."""
    m = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(m), X])
    theta = (
        np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        @ X_with_intercept.T
        @ y
    )
    return theta


def compute_mse(y_true, y_pred):
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


# =============================================================================
# Basic Functionality Tests - Batch Gradient Descent
# =============================================================================


def test_batch_gd_simple_1d():
    """
    Test batch GD with simple one-dimensional data.

    This test uses a simple dataset where the relationship is y = 2x + 1.
    Gradient descent should converge close to the closed-form solution.
    """
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)  # y = 2x + 1

    # Fit using gradient descent
    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model.fit(X, y)
    predictions = model.predict(X)

    # Compare with closed-form solution
    theta_closed = closed_form_solution(X, y)

    # Coefficients should be close
    np.testing.assert_allclose(model.theta, theta_closed, rtol=1e-3, atol=1e-3)

    # Predictions should be very accurate
    np.testing.assert_allclose(predictions, y, rtol=1e-2, atol=1e-2)


def test_batch_gd_multiple_features():
    """
    Test batch GD with multiple features.

    Verifies that the implementation correctly handles multi-dimensional inputs.
    """
    np.random.seed(42)
    m, d = 100, 3
    X = np.random.randn(m, d)
    true_theta = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_theta + 0.1 * np.random.randn(m)

    # Standardize features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=2000, verbose=False
    )
    model.fit(X_scaled, y)

    # Compare with closed-form on scaled data
    theta_closed = closed_form_solution(X_scaled, y)

    np.testing.assert_allclose(model.theta, theta_closed, rtol=1e-2, atol=1e-2)


def test_batch_gd_prediction_on_new_data():
    """
    Test that batch GD can predict on unseen data.
    """
    X_train = np.array([[1], [2], [3], [4]], dtype=float)
    y_train = np.array([2, 4, 6, 8], dtype=float)  # y = 2x
    X_test = np.array([[5], [6], [0]], dtype=float)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Expected predictions (approximately)
    expected = np.array([10, 12, 0], dtype=float)
    np.testing.assert_allclose(predictions, expected, rtol=0.05, atol=0.1)


# =============================================================================
# Convergence Tests
# =============================================================================


def test_batch_gd_loss_decreases():
    """
    Test that loss consistently decreases during training.

    The loss should be monotonically decreasing for batch gradient descent
    with appropriate learning rate.
    """
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * np.random.randn(50)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=500, verbose=False
    )
    model.fit(X_scaled, y)

    # Check that loss generally decreases
    loss_history = model.loss_history
    assert len(loss_history) > 0, "Loss history should not be empty"

    # Allow for small increases due to numerical issues, but overall trend should be down
    assert loss_history[-1] < loss_history[0], (
        "Final loss should be less than initial loss"
    )

    # Check that most consecutive pairs show decrease
    decreases = sum(
        loss_history[i + 1] <= loss_history[i]
        for i in range(len(loss_history) - 1)
    )
    assert decreases > 0.9 * len(loss_history), (
        "Loss should decrease in most iterations"
    )


def test_batch_gd_convergence_tolerance():
    """
    Test that training stops when convergence tolerance is met.
    """
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=10000, tolerance=1e-6, verbose=False
    )
    model.fit(X, y)

    # Should converge before max iterations
    assert len(model.loss_history) < 10000, (
        "Should converge before max iterations"
    )


def test_learning_rate_too_large():
    """
    Test that very large learning rate causes divergence or oscillation.

    This test verifies that the implementation can handle (or detect)
    inappropriate learning rates.
    """
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)

    # Use a very large learning rate
    model = LinearRegressionGD(
        learning_rate=10.0, num_iterations=100, verbose=False
    )
    model.fit(X, y)

    # Loss should either increase or oscillate wildly
    loss_history = model.loss_history

    # Check if loss explodes or doesn't converge properly
    # (either NaN, very large values, or non-decreasing pattern)
    if not np.any(np.isnan(loss_history)) and not np.any(
        np.isinf(loss_history)
    ):
        # If no NaN/Inf, check that it doesn't converge well
        final_loss = loss_history[-1]
        initial_loss = loss_history[0]
        # With bad learning rate, final loss shouldn't be much better
        assert final_loss > 0.1 * initial_loss or final_loss > 10


# =============================================================================
# Edge Cases
# =============================================================================


def test_batch_gd_perfect_fit():
    """
    Test with data that lies perfectly on a line.

    The model should achieve near-zero loss.
    """
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)  # Perfect: y = 2x + 1

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=2000, verbose=False
    )
    model.fit(X, y)
    predictions = model.predict(X)

    # Should achieve very small MSE
    mse = compute_mse(y, predictions)
    assert mse < 1e-4, f"MSE should be very small for perfect fit, got {mse}"


def test_batch_gd_zero_features():
    """
    Test with zero-dimensional features (intercept only).

    Model should predict the mean of y.
    """
    X = np.empty((5, 0), dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model.fit(X, y)
    predictions = model.predict(X)

    # Should predict approximately the mean
    expected_mean = np.mean(y)
    np.testing.assert_allclose(predictions, expected_mean, rtol=0.01, atol=0.01)


def test_batch_gd_negative_values():
    """
    Test with negative input values and targets.
    """
    X = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
    y = np.array([-5, -3, -1, 1, 3], dtype=float)  # y = 2x - 1

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model.fit(X, y)
    predictions = model.predict(X)

    # Should fit well
    np.testing.assert_allclose(predictions, y, rtol=0.05, atol=0.1)


# =============================================================================
# Mathematical Properties
# =============================================================================


def test_batch_gd_recovers_true_parameters():
    """
    Test that gradient descent recovers known true parameters.
    """
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    true_slope = 3.0
    true_intercept = 2.0
    y = true_slope * X.flatten() + true_intercept

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=2000, verbose=False
    )
    model.fit(X, y)

    # theta should be [intercept, slope]
    expected_theta = np.array([true_intercept, true_slope])
    np.testing.assert_allclose(
        model.theta, expected_theta, rtol=1e-2, atol=1e-2
    )


# =============================================================================
# Feature Scaling Tests
# =============================================================================


def test_feature_scaling_improves_convergence():
    """
    Test that feature scaling dramatically improves convergence speed.

    This test creates data with features at very different scales.
    """
    np.random.seed(42)
    m = 100
    X_unscaled = np.column_stack(
        [
            np.random.randn(m) * 1000,  # Large scale
            np.random.randn(m) * 1,  # Small scale
        ]
    )
    y = (
        2 * X_unscaled[:, 0] / 1000
        + 5 * X_unscaled[:, 1]
        + 3
        + np.random.randn(m)
    )

    # Without scaling - needs tiny learning rate
    model_unscaled = LinearRegressionGD(
        learning_rate=0.00001, num_iterations=5000, verbose=False
    )
    model_unscaled.fit(X_unscaled, y)

    # With scaling - can use larger learning rate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)

    model_scaled = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model_scaled.fit(X_scaled, y)

    # Scaled version should converge much faster
    # (fewer iterations to reach similar loss)
    assert len(model_scaled.loss_history) < len(model_unscaled.loss_history)

    # Scaled version should achieve lower final loss
    assert model_scaled.loss_history[-1] < model_unscaled.loss_history[-1] * 1.5


# =============================================================================
# Stochastic Gradient Descent Tests
# =============================================================================


def test_sgd_converges_to_similar_solution():
    """
    Test that SGD converges to a solution similar to batch GD.

    SGD won't match exactly due to its stochastic nature, but should
    be close to the batch GD solution.
    """
    np.random.seed(42)
    m, d = 200, 3
    X = np.random.randn(m, d)
    true_theta = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_theta + 0.1 * np.random.randn(m)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Batch GD
    model_batch = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model_batch.fit(X_scaled, y)

    # SGD
    model_sgd = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=50, verbose=False
    )
    model_sgd.fit(X_scaled, y)

    # Parameters should be reasonably close
    np.testing.assert_allclose(
        model_sgd.theta, model_batch.theta, rtol=0.1, atol=0.1
    )


def test_sgd_prediction():
    """
    Test that SGD can make reasonable predictions.
    """
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)
    X_test = np.array([[5], [6]], dtype=float)

    model = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=100, verbose=False
    )
    model.fit(X, y)
    predictions = model.predict(X_test)

    # Should predict approximately [10, 12]
    expected = np.array([10, 12], dtype=float)
    np.testing.assert_allclose(predictions, expected, rtol=0.1, atol=0.5)


def test_sgd_loss_general_decrease():
    """
    Test that SGD loss generally decreases over epochs.

    SGD loss will be noisy, but the overall trend should be downward.
    """
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * np.random.randn(100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=50, verbose=False
    )
    model.fit(X_scaled, y)

    # Final loss should be less than initial loss
    assert len(model.loss_history) > 0
    assert model.loss_history[-1] < model.loss_history[0]


# =============================================================================
# Mini-Batch Gradient Descent Tests
# =============================================================================


def test_minibatch_gd_converges():
    """
    Test that mini-batch GD converges to a good solution.
    """
    np.random.seed(42)
    m, d = 200, 3
    X = np.random.randn(m, d)
    true_theta = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_theta + 0.1 * np.random.randn(m)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegressionMiniBatchGD(
        learning_rate=0.1, num_epochs=50, batch_size=32, verbose=False
    )
    model.fit(X_scaled, y)

    # Compare with closed-form
    theta_closed = closed_form_solution(X_scaled, y)
    np.testing.assert_allclose(model.theta, theta_closed, rtol=0.05, atol=0.05)


def test_minibatch_gd_different_batch_sizes():
    """
    Test mini-batch GD with different batch sizes.

    All should converge to similar solutions.
    """
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * np.random.randn(100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    batch_sizes = [16, 32, 64]
    models = []

    for batch_size in batch_sizes:
        model = LinearRegressionMiniBatchGD(
            learning_rate=0.1,
            num_epochs=50,
            batch_size=batch_size,
            verbose=False,
        )
        model.fit(X_scaled, y)
        models.append(model)

    # All models should have similar parameters
    for i in range(len(models) - 1):
        np.testing.assert_allclose(
            models[i].theta, models[i + 1].theta, rtol=0.1, atol=0.1
        )


def test_minibatch_prediction():
    """
    Test that mini-batch GD can make predictions.
    """
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    X_test = np.array([[9], [10]], dtype=float)

    model = LinearRegressionMiniBatchGD(
        learning_rate=0.1, num_epochs=100, batch_size=4, verbose=False
    )
    model.fit(X, y)
    predictions = model.predict(X_test)

    expected = np.array([18, 20], dtype=float)
    np.testing.assert_allclose(predictions, expected, rtol=0.1, atol=0.5)


# =============================================================================
# Comparison Tests - All Three Methods
# =============================================================================


def test_all_methods_converge_similarly():
    """
    Test that batch GD, SGD, and mini-batch GD all converge to similar solutions.
    """
    np.random.seed(42)
    m, d = 200, 3
    X = np.random.randn(m, d)
    true_theta = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_theta + 0.1 * np.random.randn(m)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Batch GD
    model_batch = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model_batch.fit(X_scaled, y)

    # SGD
    model_sgd = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=50, verbose=False
    )
    model_sgd.fit(X_scaled, y)

    # Mini-batch GD
    model_minibatch = LinearRegressionMiniBatchGD(
        learning_rate=0.1, num_epochs=50, batch_size=32, verbose=False
    )
    model_minibatch.fit(X_scaled, y)

    # All should be close to each other
    np.testing.assert_allclose(
        model_batch.theta, model_sgd.theta, rtol=0.1, atol=0.1
    )
    np.testing.assert_allclose(
        model_batch.theta, model_minibatch.theta, rtol=0.05, atol=0.05
    )


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize("n_samples,n_features", [(50, 1), (100, 2), (200, 5)])
def test_batch_gd_various_sizes(n_samples, n_features):
    """
    Test batch GD with various data sizes and dimensions.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model.fit(X_scaled, y)

    # Compare with closed-form
    theta_closed = closed_form_solution(X_scaled, y)
    np.testing.assert_allclose(model.theta, theta_closed, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("learning_rate", [0.01, 0.05, 0.1, 0.5])
def test_batch_gd_various_learning_rates(learning_rate):
    """
    Test batch GD with various learning rates.

    All reasonable learning rates should converge to similar solutions.
    """
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * np.random.randn(50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegressionGD(
        learning_rate=learning_rate, num_iterations=2000, verbose=False
    )
    model.fit(X_scaled, y)

    # Should converge reasonably well
    theta_closed = closed_form_solution(X_scaled, y)
    np.testing.assert_allclose(model.theta, theta_closed, rtol=0.1, atol=0.1)
