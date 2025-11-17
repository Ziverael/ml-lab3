"""
Simple test suite for Gradient Descent Linear Regression implementations.

Tests the three main implementations:
- LinearRegressionGD (Batch Gradient Descent)
- LinearRegressionSGD (Stochastic Gradient Descent)
- LinearRegressionMiniBatchGD (Mini-Batch Gradient Descent)

Compares against the closed-form solution for validation.
"""

import numpy as np
import pytest

from src.linear_regression_gd import LinearRegressionGD
from src.linear_regression_sgd import LinearRegressionSGD
from src.linear_regression_minibatch_gd import LinearRegressionMiniBatchGD


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


# =============================================================================
# Batch Gradient Descent Tests
# =============================================================================


def test_batch_gd_one_dimensional():
    """Test batch GD on simple 1D data."""
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)  # y = 2x + 1

    X_test = np.array([6, 7, 0]).reshape((3, 1)).astype(float)
    expected_closed = closed_form_solution(X, y)

    # Closed form predictions
    X_test_with_intercept = np.column_stack([np.ones(3), X_test])
    expected = X_test_with_intercept @ expected_closed

    # Gradient descent
    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model.fit(X, y)
    actual = model.predict(X_test)

    assert list(actual) == pytest.approx(list(expected), rel=1e-2)


def test_batch_gd_multi_dimensional():
    """Test batch GD on multi-dimensional data."""
    X = (
        np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5])
        .reshape((4, 3))
        .astype(float)
    )
    y = np.array([2, 5, 3, 8]).astype(float)

    X_test = (
        np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7])
        .reshape((4, 3))
        .astype(float)
    )

    expected_closed = closed_form_solution(X, y)
    X_test_with_intercept = np.column_stack([np.ones(4), X_test])
    expected = X_test_with_intercept @ expected_closed

    model = LinearRegressionGD(
        learning_rate=0.01, num_iterations=2000, verbose=False
    )
    model.fit(X, y)
    actual = model.predict(X_test)

    assert list(actual) == pytest.approx(list(expected), rel=1e-1)


def test_batch_gd_perfect_fit():
    """Test batch GD on perfectly linear data."""
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model.fit(X, y)
    predictions = model.predict(X)

    # Should predict training data almost exactly
    assert list(predictions) == pytest.approx(list(y), rel=1e-2)


def test_batch_gd_recovers_parameters():
    """Test that batch GD recovers the true parameters."""
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    true_slope = 3.0
    true_intercept = 2.0
    y = true_slope * X.flatten() + true_intercept

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=2000, verbose=False
    )
    model.fit(X, y)

    # theta = [intercept, slope]
    expected_theta = [true_intercept, true_slope]
    assert list(model.theta) == pytest.approx(expected_theta, rel=1e-2)


def test_batch_gd_loss_decreases():
    """Test that loss decreases during training."""
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)

    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=100, verbose=False
    )
    model.fit(X, y)

    # Loss should decrease
    assert model.loss_history[-1] < model.loss_history[0]


# =============================================================================
# Stochastic Gradient Descent Tests
# =============================================================================


def test_sgd_one_dimensional():
    """Test SGD on simple 1D data."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((8, 1)).astype(float)
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16]).astype(float)  # y = 2x

    X_test = np.array([9, 10]).reshape((2, 1)).astype(float)
    expected_closed = closed_form_solution(X, y)
    X_test_with_intercept = np.column_stack([np.ones(2), X_test])
    expected = X_test_with_intercept @ expected_closed

    model = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=100, verbose=False
    )
    model.fit(X, y)
    actual = model.predict(X_test)

    # SGD is less precise, so use larger tolerance
    assert list(actual) == pytest.approx(list(expected), rel=0.1)


def test_sgd_multi_dimensional():
    """Test SGD on multi-dimensional data."""
    X = (
        np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5, 1, 1, 2, 3])
        .reshape((4, 4))
        .astype(float)
    )
    y = np.array([2, 5, 3, 8]).astype(float)

    X_test = np.array([1, 0, 0, 0, 0, 1, 0, 0]).reshape((2, 4)).astype(float)

    expected_closed = closed_form_solution(X, y)
    X_test_with_intercept = np.column_stack([np.ones(2), X_test])
    expected = X_test_with_intercept @ expected_closed

    model = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=200, verbose=False
    )
    model.fit(X, y)
    actual = model.predict(X_test)

    assert list(actual) == pytest.approx(list(expected), rel=0.2)


def test_sgd_general_loss_decrease():
    """Test that SGD loss generally decreases."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * np.random.randn(100)

    model = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=50, verbose=False
    )
    model.fit(X, y)

    # Final loss should be less than initial (general trend)
    assert model.loss_history[-1] < model.loss_history[0]


# =============================================================================
# Mini-Batch Gradient Descent Tests
# =============================================================================


def test_minibatch_gd_one_dimensional():
    """Test mini-batch GD on simple 1D data."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((8, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11, 13, 15, 17]).astype(float)  # y = 2x + 1

    X_test = np.array([9, 10, 0]).reshape((3, 1)).astype(float)
    expected_closed = closed_form_solution(X, y)
    X_test_with_intercept = np.column_stack([np.ones(3), X_test])
    expected = X_test_with_intercept @ expected_closed

    model = LinearRegressionMiniBatchGD(
        learning_rate=0.1, num_epochs=100, batch_size=4, verbose=False
    )
    model.fit(X, y)
    actual = model.predict(X_test)

    assert list(actual) == pytest.approx(list(expected), rel=0.05)


def test_minibatch_gd_multi_dimensional():
    """Test mini-batch GD on multi-dimensional data."""
    X = (
        np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5])
        .reshape((4, 3))
        .astype(float)
    )
    y = np.array([2, 5, 3, 8]).astype(float)

    X_test = np.array([1, 0, 0, 0, 1, 0]).reshape((2, 3)).astype(float)

    expected_closed = closed_form_solution(X, y)
    X_test_with_intercept = np.column_stack([np.ones(2), X_test])
    expected = X_test_with_intercept @ expected_closed

    model = LinearRegressionMiniBatchGD(
        learning_rate=0.05, num_epochs=200, batch_size=2, verbose=False
    )
    model.fit(X, y)
    actual = model.predict(X_test)

    assert list(actual) == pytest.approx(list(expected), rel=0.1)


def test_minibatch_gd_different_batch_sizes():
    """Test that different batch sizes produce similar results."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X @ np.array([1.5, -2.0]) + 3

    X_test = np.array([0, 0, 1, 1]).reshape((2, 2)).astype(float)

    # Train with batch size 16
    model1 = LinearRegressionMiniBatchGD(
        learning_rate=0.1, num_epochs=50, batch_size=16, verbose=False
    )
    model1.fit(X, y)
    pred1 = model1.predict(X_test)

    # Train with batch size 32
    model2 = LinearRegressionMiniBatchGD(
        learning_rate=0.1, num_epochs=50, batch_size=32, verbose=False
    )
    model2.fit(X, y)
    pred2 = model2.predict(X_test)

    # Should produce similar predictions
    assert list(pred1) == pytest.approx(list(pred2), rel=0.1)


# =============================================================================
# Comparison Tests
# =============================================================================


def test_all_methods_similar_results():
    """Test that all three methods produce similar results."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([1.0, 2.0, -1.5]) + 3 + 0.1 * np.random.randn(100)

    X_test = np.array([0, 0, 0, 1, 1, 1]).reshape((2, 3)).astype(float)

    # Batch GD
    model_batch = LinearRegressionGD(
        learning_rate=0.1, num_iterations=1000, verbose=False
    )
    model_batch.fit(X, y)
    pred_batch = model_batch.predict(X_test)

    # SGD
    model_sgd = LinearRegressionSGD(
        learning_rate=0.01, num_epochs=50, verbose=False
    )
    model_sgd.fit(X, y)
    pred_sgd = model_sgd.predict(X_test)

    # Mini-batch GD
    model_minibatch = LinearRegressionMiniBatchGD(
        learning_rate=0.1, num_epochs=50, batch_size=20, verbose=False
    )
    model_minibatch.fit(X, y)
    pred_minibatch = model_minibatch.predict(X_test)

    # All should be reasonably close
    assert list(pred_batch) == pytest.approx(list(pred_sgd), rel=0.15)
    assert list(pred_batch) == pytest.approx(list(pred_minibatch), rel=0.1)


def test_convergence_to_closed_form():
    """Test that batch GD converges close to closed-form solution."""
    X = np.array([1, 3, 2, 5, 4, 6]).reshape((6, 1)).astype(float)
    y = np.array([2, 5, 3, 8, 7, 10]).astype(float)

    # Closed form
    theta_closed = closed_form_solution(X, y)

    # Batch GD
    model = LinearRegressionGD(
        learning_rate=0.1, num_iterations=2000, verbose=False
    )
    model.fit(X, y)

    # Parameters should be close
    assert list(model.theta) == pytest.approx(list(theta_closed), rel=0.05)
