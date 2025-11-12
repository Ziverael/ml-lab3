"""
Test suite for Ridge Regression with Gradient Descent implementation.

Tests the RidgeRegressionGD class against sklearn's Ridge regression
to ensure correctness of the L2-regularized gradient descent implementation.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.ridge_regression_gd import RidgeRegressionGD


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_ridge_gd_one_dimensional():
    """Test Ridge GD on simple 1D data."""
    X = np.array([1, 3, 2, 5]).reshape((4, 1)).astype(float)
    y = np.array([2, 5, 3, 8]).astype(float)
    
    alpha = 0.3
    X_test = np.array([1, 2, 10]).reshape((3, 1)).astype(float)
    
    # sklearn Ridge
    expected = Ridge(alpha=alpha).fit(X, y).predict(X_test)
    
    # Our implementation
    model = RidgeRegressionGD(alpha=alpha, learning_rate=0.1, num_iterations=2000, verbose=False)
    actual = model.fit(X, y).predict(X_test)
    
    assert list(actual) == pytest.approx(list(expected), rel=1e-2)


def test_ridge_gd_multi_dimensional():
    """Test Ridge GD on multi-dimensional data."""
    X = np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5]).reshape((4, 3)).astype(float)
    y = np.array([2, 5, 3, 8]).astype(float)
    
    X_test = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3]).reshape((5, 3)).astype(float)
    
    alpha = 0.4
    
    # sklearn Ridge
    expected = Ridge(alpha=alpha).fit(X, y).predict(X_test)
    
    # Our implementation
    model = RidgeRegressionGD(alpha=alpha, learning_rate=0.1, num_iterations=2000, verbose=False)
    actual = model.fit(X, y).predict(X_test)
    
    assert list(actual) == pytest.approx(list(expected), rel=0.05)


def test_ridge_gd_zero_alpha():
    """Test that Ridge with alpha=0 behaves like ordinary linear regression."""
    X = np.array([1, 3, 2, 5]).reshape((4, 1)).astype(float)
    y = np.array([2, 5, 3, 8]).astype(float)
    
    alpha = 0
    X_test = np.array([1, 2, 10]).reshape((3, 1)).astype(float)
    
    # sklearn Ridge with alpha=0
    expected = Ridge(alpha=alpha).fit(X, y).predict(X_test)
    
    # Our implementation with alpha=0
    model = RidgeRegressionGD(alpha=alpha, learning_rate=0.1, num_iterations=2000, verbose=False)
    actual = model.fit(X, y).predict(X_test)
    
    assert list(actual) == pytest.approx(list(expected), rel=1e-2)


# =============================================================================
# Regularization Effect Tests
# =============================================================================


def test_ridge_reduces_coefficient_magnitude():
    """Test that Ridge regression reduces coefficient magnitudes compared to OLS."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([3.0, -2.0, 1.5]) + 0.1 * np.random.randn(50)
    
    # Standardize for fair comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Without regularization (alpha=0)
    model_ols = RidgeRegressionGD(alpha=0, learning_rate=0.1, num_iterations=2000, verbose=False)
    model_ols.fit(X_scaled, y)
    
    # With regularization (alpha=1.0)
    model_ridge = RidgeRegressionGD(alpha=1.0, learning_rate=0.1, num_iterations=2000, verbose=False)
    model_ridge.fit(X_scaled, y)
    
    # Ridge coefficients (excluding intercept) should have smaller magnitude
    ols_coef_magnitude = np.linalg.norm(model_ols.theta[1:])
    ridge_coef_magnitude = np.linalg.norm(model_ridge.theta[1:])
    
    assert ridge_coef_magnitude < ols_coef_magnitude


def test_ridge_handles_multicollinearity():
    """Test that Ridge can handle highly correlated features."""
    np.random.seed(42)
    X1 = np.random.randn(50, 1)
    X2 = X1 + 0.01 * np.random.randn(50, 1)  # Highly correlated with X1
    X = np.column_stack([X1, X2])
    y = X1.flatten() + 0.1 * np.random.randn(50)
    
    # Ridge should handle this (doesn't require matrix inversion)
    model = RidgeRegressionGD(alpha=0.5, learning_rate=0.1, num_iterations=2000, verbose=False)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Should produce reasonable predictions
    mse = np.mean((predictions - y) ** 2)
    assert mse < 1.0  # Should fit reasonably well


def test_higher_alpha_more_regularization():
    """Test that higher alpha values lead to more regularization."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([3.0, -2.0, 1.5]) + 0.1 * np.random.randn(50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Different alpha values
    model_weak = RidgeRegressionGD(alpha=0.1, learning_rate=0.1, num_iterations=2000, verbose=False)
    model_weak.fit(X_scaled, y)
    
    model_strong = RidgeRegressionGD(alpha=10.0, learning_rate=0.1, num_iterations=2000, verbose=False)
    model_strong.fit(X_scaled, y)
    
    # Higher alpha should lead to smaller coefficient magnitudes
    weak_magnitude = np.linalg.norm(model_weak.theta[1:])
    strong_magnitude = np.linalg.norm(model_strong.theta[1:])
    
    assert strong_magnitude < weak_magnitude


# =============================================================================
# Mathematical Properties Tests
# =============================================================================


def test_ridge_intercept_not_regularized():
    """Test that the intercept term is not regularized."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X @ np.array([1.5, -2.0]) + 5.0 + 0.1 * np.random.randn(50)  # Intercept = 5.0
    
    # With very high regularization
    model = RidgeRegressionGD(alpha=100.0, learning_rate=0.1, num_iterations=2000, verbose=False)
    model.fit(X, y)
    
    # Intercept should still be close to 5.0 (not shrunk to zero)
    # while slopes should be heavily regularized
    assert abs(model.theta[0] - 5.0) < 1.0  # Intercept not too affected
    assert np.linalg.norm(model.theta[1:]) < 1.0  # Slopes heavily regularized


def test_ridge_loss_decreases():
    """Test that loss decreases during training."""
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)
    
    model = RidgeRegressionGD(alpha=0.5, learning_rate=0.1, num_iterations=500, verbose=False)
    model.fit(X, y)
    
    # Loss should decrease
    assert len(model.loss_history) > 0
    assert model.loss_history[-1] < model.loss_history[0]


# =============================================================================
# Comparison with sklearn Tests
# =============================================================================


def test_ridge_matches_sklearn_simple():
    """Test that our implementation closely matches sklearn on simple data."""
    X = np.array([[1], [2], [3], [4], [5]]).astype(float)
    y = np.array([2, 4, 5, 4, 5]).astype(float)
    
    alpha = 1.0
    
    # sklearn
    sklearn_model = Ridge(alpha=alpha)
    sklearn_model.fit(X, y)
    sklearn_pred = sklearn_model.predict(X)
    
    # Our implementation
    model = RidgeRegressionGD(alpha=alpha, learning_rate=0.1, num_iterations=3000, verbose=False)
    model.fit(X, y)
    our_pred = model.predict(X)
    
    # Predictions should be very close
    assert list(our_pred) == pytest.approx(list(sklearn_pred), rel=0.05)


def test_ridge_matches_sklearn_random():
    """Test against sklearn on random data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    alpha = 1.0
    
    # sklearn
    sklearn_model = Ridge(alpha=alpha)
    sklearn_model.fit(X_scaled, y_scaled)
    
    # Our implementation
    model = RidgeRegressionGD(alpha=alpha, learning_rate=0.1, num_iterations=2000, verbose=False)
    model.fit(X_scaled, y_scaled)
    
    # Parameters should be close
    # Note: sklearn doesn't expose theta in the same format, so compare predictions
    X_test = np.random.randn(10, 5)
    X_test_scaled = scaler_X.transform(X_test)
    
    sklearn_pred = sklearn_model.predict(X_test_scaled)
    our_pred = model.predict(X_test_scaled)
    
    assert list(our_pred) == pytest.approx(list(sklearn_pred), rel=0.1)


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_ridge_with_negative_values():
    """Test Ridge GD with negative inputs and targets."""
    X = np.array([[-2], [-1], [0], [1], [2]]).astype(float)
    y = np.array([-5, -3, -1, 1, 3]).astype(float)
    
    alpha = 0.5
    
    model = RidgeRegressionGD(alpha=alpha, learning_rate=0.1, num_iterations=2000, verbose=False)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Should predict reasonably well
    assert list(predictions) == pytest.approx(list(y), rel=0.1)


def test_ridge_perfect_fit_with_regularization():
    """Test that Ridge with regularization doesn't achieve perfect fit even on linear data."""
    X = np.array([[1], [2], [3], [4], [5]]).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)  # Perfect line: y = 2x + 1
    
    # With strong regularization
    model = RidgeRegressionGD(alpha=10.0, learning_rate=0.1, num_iterations=2000, verbose=False)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Should NOT achieve perfect fit due to regularization
    mse = np.mean((predictions - y) ** 2)
    assert mse > 0.01  # Some error due to regularization


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 5.0])
def test_ridge_various_alphas(alpha):
    """Test Ridge GD with various alpha values."""
    X = np.array([1, 3, 2, 5, 4]).reshape((5, 1)).astype(float)
    y = np.array([2, 5, 3, 8, 7]).astype(float)
    
    # sklearn
    sklearn_model = Ridge(alpha=alpha)
    sklearn_model.fit(X, y)
    sklearn_pred = sklearn_model.predict(X)
    
    # Our implementation
    model = RidgeRegressionGD(alpha=alpha, learning_rate=0.1, num_iterations=3000, verbose=False)
    model.fit(X, y)
    our_pred = model.predict(X)
    
    assert list(our_pred) == pytest.approx(list(sklearn_pred), rel=0.1)
