# Gradient Descent for Linear Regression: An Alternative to the Closed-Form Solution

**Author:** Tutorial Series  
**Date:** November 3, 2025  
**Topics:** gradient-descent, linear-regression, optimization, machine-learning

## Introduction

In previous discussions of multiple linear regression, you've learned about the closed-form solution for estimating model parameters. Given a design matrix $\mathbf{X}$ and target vector $\mathbf{y}$, the optimal parameters $\boldsymbol{\theta}$ can be computed directly using the Normal Equation:

$\boldsymbol{\theta}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

This elegant solution provides the exact optimal parameters in a single computation. However, in many real-world scenarios, this closed-form approach becomes impractical or even impossible to use. In this tutorial, we'll explore **gradient descent**, an iterative optimization algorithm that provides a powerful alternative for estimating linear regression parameters, especially when the closed-form solution struggles.

## When Does the Closed-Form Solution Fail?

Before diving into gradient descent, it's important to understand the limitations of the closed-form solution:

### 1. **Computational Complexity**
The closed-form solution requires computing $(\mathbf{X}^T\mathbf{X})^{-1}$, which involves:
- Matrix multiplication: $O(n \cdot d^2)$ for $n$ samples and $d$ features
- Matrix inversion: $O(d^3)$

For large numbers of features (e.g., $d > 10,000$), this becomes prohibitively expensive.

### 2. **Memory Requirements**
The matrix $\mathbf{X}^T\mathbf{X}$ is $d \times d$. For $d = 100,000$ features, this requires storing 10 billion floating-point numbers—approximately 40GB of memory!

### 3. **Non-Invertible Matrices**
When features are perfectly correlated (multicollinearity) or when $d > n$ (more features than samples), the matrix $\mathbf{X}^T\mathbf{X}$ becomes singular and cannot be inverted.

### 4. **Online Learning**
The closed-form solution requires all data to be available at once. It cannot easily incorporate new data points without recomputing everything from scratch.

Gradient descent elegantly addresses all of these limitations.

## The Core Idea: Gradient Descent

Gradient descent is an iterative optimization algorithm that finds the minimum of a function by repeatedly taking steps in the direction of steepest descent. Think of it as a hiker trying to reach the bottom of a valley in thick fog—they can't see the bottom, but they can feel which direction is downhill at their current position.

### The Loss Function

For linear regression, we want to minimize the **Mean Squared Error (MSE)** loss function:

$$J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

where:
- $m$ is the number of training examples
- $h_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta}^T\mathbf{x}$ is our hypothesis function
- $\mathbf{x}^{(i)}$ is the $i$-th input feature vector
- $y^{(i)}$ is the $i$-th target value
- The factor $\frac{1}{2}$ is included for mathematical convenience (it cancels when we take derivatives)

### The Gradient

The **gradient** of $J(\boldsymbol{\theta})$ with respect to $\boldsymbol{\theta}$ is a vector of partial derivatives that points in the direction of steepest *increase* of the function:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \begin{bmatrix}
\frac{\partial J}{\partial \theta_0} \\
\frac{\partial J}{\partial \theta_1} \\
\vdots \\
\frac{\partial J}{\partial \theta_d}
\end{bmatrix}$$

For our MSE loss function, this gradient has a beautiful closed form:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \frac{1}{m}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

**Derivation:** For those interested in the mathematics, here's how we arrive at this result:

$$\frac{\partial J}{\partial \theta_j} = \frac{\partial}{\partial \theta_j}\left[\frac{1}{2m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)})^2\right]$$

$$= \frac{1}{m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)}) \cdot \frac{\partial}{\partial \theta_j}h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})$$

$$= \frac{1}{m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

In vectorized form, this becomes: $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \frac{1}{m}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$

### The Update Rule

To minimize $J(\boldsymbol{\theta})$, we repeatedly update our parameters by taking a step in the *opposite* direction of the gradient:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}^{(t)})$$

where:
- $t$ is the iteration number
- $\alpha$ is the **learning rate**, a positive scalar that controls the step size

## The Learning Rate: A Critical Hyperparameter

The learning rate $\alpha$ is one of the most important hyperparameters in gradient descent. It determines how large a step we take in the direction of steepest descent.

### Choosing the Right Learning Rate

**If $\alpha$ is too small:**
- The algorithm converges very slowly
- Many iterations are needed to reach the minimum
- Training takes excessive time

**If $\alpha$ is too large:**
- The algorithm may overshoot the minimum
- Parameters can diverge, making the loss function increase
- The algorithm may oscillate or fail to converge

**The "just right" $\alpha$:**
- Converges in a reasonable number of iterations
- Makes steady progress toward the minimum
- Doesn't overshoot or oscillate excessively

### Visualizing Different Learning Rates

Below we demonstrate how different learning rates affect convergence for a simple 1D problem where $J(\theta) = (\theta - 3)^2$:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a simple quadratic loss function
def J(theta):
    return (theta - 3)**2

def gradient_J(theta):
    return 2*(theta - 3)

# Gradient descent with different learning rates
def gradient_descent_1d(alpha, num_iterations=20, theta_init=0):
    theta = theta_init
    history = [theta]
    
    for i in range(num_iterations):
        theta = theta - alpha * gradient_J(theta)
        history.append(theta)
    
    return np.array(history)

# Test different learning rates
theta_range = np.linspace(-2, 8, 100)
learning_rates = [0.1, 0.5, 0.9, 1.1]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, alpha in enumerate(learning_rates):
    ax = axes[idx]
    
    # Plot the loss function
    ax.plot(theta_range, J(theta_range), 'k-', linewidth=2, label='J(θ)')
    
    # Run gradient descent
    theta_history = gradient_descent_1d(alpha, num_iterations=20)
    
    # Plot the path
    ax.plot(theta_history, J(theta_history), 'ro-', markersize=8, 
            linewidth=1.5, alpha=0.7, label='GD Path')
    ax.plot(theta_history[0], J(theta_history[0]), 'go', 
            markersize=12, label='Start')
    ax.plot(3, 0, 'b*', markersize=15, label='Minimum')
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('J(θ)', fontsize=12)
    ax.set_title(f'Learning Rate α = {alpha}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 30)

plt.tight_layout()
plt.show()
```

**Figure 1:** Effect of different learning rates on gradient descent convergence. (Top-left) $\alpha = 0.1$: Slow but steady convergence. (Top-right) $\alpha = 0.5$: Good convergence rate. (Bottom-left) $\alpha = 0.9$: Fast convergence, slight oscillation. (Bottom-right) $\alpha = 1.1$: Too large—diverges!

## Batch Gradient Descent: The Standard Algorithm

The most straightforward implementation of gradient descent is **Batch Gradient Descent**, which uses all training examples to compute the gradient at each iteration.

### Algorithm

```
Initialize: θ randomly (or to zeros)
Repeat until convergence:
    1. Compute gradient: ∇J(θ) = (1/m)X^T(Xθ - y)
    2. Update parameters: θ := θ - α∇J(θ)
    3. Check convergence criterion
```

### Convergence Criteria

We need to decide when to stop the algorithm. Common criteria include:

1. **Maximum iterations:** Stop after a fixed number of iterations
2. **Gradient magnitude:** Stop when $||\nabla J(\boldsymbol{\theta})|| < \epsilon$
3. **Loss change:** Stop when $|J(\boldsymbol{\theta}^{(t)}) - J(\boldsymbol{\theta}^{(t-1)})| < \epsilon$

### Implementation Exercise

Now it's your turn! Below is a skeleton for a `LinearRegressionGD` class. **Your task is to implement the `fit` and `predict` methods.**

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    """Linear Regression using Batch Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, num_iterations=1000, 
                 tolerance=1e-6, verbose=True):
        """
        Parameters:
        -----------
        learning_rate : float
            The learning rate (alpha) for gradient descent
        num_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance for loss change
        verbose : bool
            Whether to print progress information
        """
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
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        return loss
    
    def _compute_gradient(self, X, y, theta):
        """Compute gradient of MSE loss"""
        m = len(y)
        predictions = X @ theta
        gradient = (1/m) * X.T @ (predictions - y)
        return gradient
    
    def fit(self, X, y):
        """
        Fit the linear regression model using gradient descent
        
        Parameters:
        -----------
        X : numpy array of shape (m, d)
            Training features (without intercept column)
        y : numpy array of shape (m,)
            Training targets
            
        Returns:
        --------
        self : object
            Returns self for method chaining
            
        TODO: Implement this method!
        Your implementation should:
        1. Add an intercept column of ones to X
        2. Initialize theta to zeros (d+1 dimensional)
        3. Run gradient descent for num_iterations:
           a. Compute the gradient using _compute_gradient
           b. Update theta using the update rule
           c. Compute and store the loss
           d. Print progress if verbose is True (every 100 iterations)
           e. Check for convergence based on loss change
        4. Return self
        """
        # YOUR CODE HERE
        raise NotImplementedError("You need to implement the fit method!")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : numpy array of shape (m, d)
            Feature matrix (without intercept column)
            
        Returns:
        --------
        predictions : numpy array of shape (m,)
            Predicted target values
            
        TODO: Implement this method!
        Your implementation should:
        1. Add an intercept column to X
        2. Compute predictions as X @ theta
        3. Return the predictions
        """
        # YOUR CODE HERE
        raise NotImplementedError("You need to implement the predict method!")
    
    def plot_loss_history(self):
        """Plot the loss function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss J(θ)', fontsize=12)
        plt.title('Loss Function vs. Iteration', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()
```

### Testing Your Implementation

Once you've implemented `fit` and `predict`, test your code with this simple example:

```python
# Generate synthetic data
np.random.seed(42)
m = 100  # number of samples
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(m)  # y = 4 + 3x + noise

# Fit using gradient descent
model_gd = LinearRegressionGD(learning_rate=0.1, num_iterations=1000, verbose=True)
model_gd.fit(X, y)

# Compare with closed-form solution
X_with_intercept = np.column_stack([np.ones(m), X])
theta_closed_form = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

print("\nGradient Descent Solution:")
print(f"θ₀ (intercept) = {model_gd.theta[0]:.4f}")
print(f"θ₁ (slope) = {model_gd.theta[1]:.4f}")

print("\nClosed-Form Solution:")
print(f"θ₀ (intercept) = {theta_closed_form[0]:.4f}")
print(f"θ₁ (slope) = {theta_closed_form[1]:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot data and fitted line
ax1.scatter(X, y, alpha=0.5, label='Data')
X_plot = np.array([[0], [2]])
y_pred_gd = model_gd.predict(X_plot)
ax1.plot(X_plot, y_pred_gd, 'r-', linewidth=2, label='Gradient Descent')
ax1.plot(X_plot, [theta_closed_form[0], theta_closed_form[0] + 2*theta_closed_form[1]], 
         'g--', linewidth=2, label='Closed-Form')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Linear Regression Fit', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot loss history
ax2.plot(model_gd.loss_history, linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Loss J(θ)', fontsize=12)
ax2.set_title('Loss Function Convergence', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Expected Output:** Your gradient descent solution should closely match the closed-form solution (within a small tolerance). The loss should decrease smoothly over iterations.

---

## Solution

<details>
<summary><b>Click to reveal solution</b></summary>

```python
def fit(self, X, y):
    """Fit the linear regression model using gradient descent"""
    # Add intercept term
    m, d = X.shape
    X_with_intercept = np.column_stack([np.ones(m), X])
    
    # Initialize parameters
    self.theta = np.zeros(d + 1)
    
    # Gradient descent loop
    for iteration in range(self.num_iterations):
        # Compute gradient
        gradient = self._compute_gradient(X_with_intercept, y, self.theta)
        
        # Update parameters
        self.theta = self.theta - self.learning_rate * gradient
        
        # Compute and store loss
        loss = self._compute_loss(X_with_intercept, y, self.theta)
        self.loss_history.append(loss)
        
        # Print progress
        if self.verbose and (iteration % 100 == 0 or iteration == self.num_iterations - 1):
            print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        # Check convergence
        if iteration > 0:
            loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
            if loss_change < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
    
    return self

def predict(self, X):
    """Make predictions on new data"""
    m = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(m), X])
    return X_with_intercept @ self.theta
```

</details>

---

## Multiple Regression Example

Now let's tackle a more realistic multiple regression problem:

```python
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
```

We see that gradient descent recovers parameters very close to both the true values and the closed-form solution!

## Feature Scaling: A Critical Preprocessing Step

One major challenge with gradient descent is that it can converge very slowly when features have vastly different scales. Consider a dataset where one feature ranges from 0 to 1000 while another ranges from 0 to 1. The loss function will be elongated in one direction, causing gradient descent to take a zigzag path.

### Why Feature Scaling Matters

The shape of the loss surface depends on the scale of the features. When features have different scales:
- The loss function forms elongated ellipses rather than circles
- Gradient descent takes many small steps along the long axis
- Convergence is significantly slower

### Common Scaling Methods

**1. Standardization (Z-score normalization):**
$$x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}$$

where $\mu_j$ is the mean and $\sigma_j$ is the standard deviation of feature $j$.

**2. Min-Max Scaling:**
$$x_j^{(i)} := \frac{x_j^{(i)} - \min(x_j)}{\max(x_j) - \min(x_j)}$$

This scales features to the range [0, 1].

### Demonstration

```python
# Generate data with different scales
np.random.seed(42)
m = 100
X_unscaled = np.column_stack([
    np.random.randn(m) * 1000,  # Feature 1: large scale
    np.random.randn(m) * 1      # Feature 2: small scale
])
y = 3 + 2*X_unscaled[:, 0]/1000 + 5*X_unscaled[:, 1] + np.random.randn(m)

# Fit without scaling
model_unscaled = LinearRegressionGD(learning_rate=0.00001, num_iterations=5000, verbose=False)
model_unscaled.fit(X_unscaled, y)

# Fit with scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

model_scaled = LinearRegressionGD(learning_rate=0.1, num_iterations=1000, verbose=False)
model_scaled.fit(X_scaled, y)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(model_unscaled.loss_history, linewidth=2, label='Without Scaling')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Convergence Without Feature Scaling', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(model_scaled.loss_history, linewidth=2, label='With Scaling', color='orange')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Convergence With Feature Scaling', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Iterations to converge (unscaled): {len(model_unscaled.loss_history)}")
print(f"Iterations to converge (scaled): {len(model_scaled.loss_history)}")
```

**Figure 3:** Effect of feature scaling on gradient descent convergence. (Left) Without scaling, convergence is very slow. (Right) With scaling, convergence is much faster.

## Additional Exercises

Now that you've mastered batch gradient descent, here are more exercises to deepen your understanding:

### Exercise 1: Stochastic Gradient Descent (SGD)

Implement a `LinearRegressionSGD` class in `src/linear_regression_sgd.py` that updates parameters using one training example at a time.

**Hints:**
- Shuffle the data at the beginning of each epoch
- Update parameters after each individual example
- Compute loss on the full dataset at the end of each epoch for monitoring

**Testing:** Run `pytest tests/test_linear_regression_gd.py -k sgd` to test your implementation.

### Exercise 2: Mini-Batch Gradient Descent

Implement a `LinearRegressionMiniBatchGD` class in `src/linear_regression_minibatch_gd.py` that uses mini-batches of size `batch_size`.

**Hints:**
- Divide the data into mini-batches
- Update parameters after each mini-batch
- Common batch sizes: 32, 64, 128, 256

**Testing:** Run `pytest tests/test_linear_regression_gd.py -k minibatch` to test your implementation.

### Exercise 3: Learning Rate Finder

Create a Jupyter notebook `notebooks/learning_rate_finder.ipynb` that implements an automatic learning rate finder.

**Requirements:**
1. Start with a very small learning rate (e.g., 1e-7)
2. Gradually increase it exponentially
3. Record the loss at each step
4. Plot loss vs. learning rate
5. Suggest the learning rate where loss decreases most rapidly

**Deliverable:** A notebook with visualizations showing:
- Loss vs. learning rate curve
- Recommended learning rate range
- Comparison of convergence with different learning rates

**Validation:** Your notebook should demonstrate that:
- Very small learning rates → slow convergence
- Optimal learning rate → fast, stable convergence
- Very large learning rates → divergence or oscillation

### Exercise 4: Momentum

Create a notebook `notebooks/momentum.ipynb` implementing gradient descent with momentum in a class `LinearRegressionMomentum`.

**Algorithm:**
$v_t = \beta v_{t-1} + \nabla J(\boldsymbol{\theta}_t)$
$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha v_t$

where $\beta = 0.9$ is the momentum coefficient.

**Deliverable:** A notebook demonstrating:
- Implementation of momentum
- Comparison with standard gradient descent on the same data
- Visualization showing momentum helps escape local plateaus
- Convergence speed comparison

**Expected Results:**
- Momentum should converge faster than vanilla GD
- Loss curves should show smoother convergence
- Parameters should be similar to closed-form solution

### Exercise 5: Early Stopping

Create a notebook `notebooks/early_stopping.ipynb` implementing early stopping to prevent overfitting.

**Requirements:**
1. Split data into training (80%) and validation (20%) sets
2. Monitor both training and validation loss during training
3. Stop when validation loss stops improving for N consecutive epochs (patience parameter)
4. Save the best model (lowest validation loss)
5. Implement a modified gradient descent class that tracks both losses

**Implementation Note:** You'll need to modify your gradient descent class to:
- Accept a validation set `(X_val, y_val)` in the `fit` method
- Store both `train_loss_history` and `val_loss_history`
- Implement early stopping logic based on validation loss

**Deliverable:** A notebook showing:
- Training and validation loss curves on the same plot
- Point where early stopping occurs (marked on the plot)
- Comparison of final models with/without early stopping
- Demonstration that early stopping prevents overfitting

**Suggested Test Case:**
- Generate polynomial features to create overparameterized model
- Add noise to target variable
- Show that:
  - Without early stopping: training loss ↓, validation loss ↑ (overfitting)
  - With early stopping: both losses converge, stopping at optimal point

**Expected Visualization:**
```
Loss
 ^
 |     Training Loss
 |     Validation Loss
 |   \                    /-- validation starts increasing
 |    \                  /
 |     \_______________/  ← early stopping point
 |                     \_____ training continues decreasing
 |
 +---------------------------------> Epochs
```

### Exercise 6: Ridge Regression with Gradient Descent

Create `src/ridge_regression_gd.py` implementing L2-regularized linear regression using gradient descent.

**Algorithm:**
The gradient becomes:
$\nabla J(\boldsymbol{\theta}) = \frac{1}{m}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) + \frac{\lambda}{m}\boldsymbol{\theta}$

**Important:** Don't regularize the intercept term $\theta_0$!

**Deliverable:** 
- Implementation in `src/ridge_regression_gd.py`
- Notebook `notebooks/ridge_regression.ipynb` demonstrating:
  - Comparison with `sklearn.linear_model.Ridge`
  - Effect of different $\lambda$ values
  - Handling of multicollinearity
  - Comparison with unregularized linear regression

**Testing:** Create `tests/test_ridge_regression_gd.py` with tests comparing against sklearn's Ridge regression.

---

## Exercise Summary

| Exercise | Type | File Location | Testing Method |
|----------|------|---------------|----------------|
| 1. SGD | Implementation | `src/linear_regression_sgd.py` | pytest tests |
| 2. Mini-Batch GD | Implementation | `src/linear_regression_minibatch_gd.py` | pytest tests |
| 3. Learning Rate Finder | Notebook | `notebooks/learning_rate_finder.ipynb` | Visual validation |
| 4. Momentum | Notebook | `notebooks/momentum.ipynb` | Visual validation |
| 5. Early Stopping | Notebook | `notebooks/early_stopping.ipynb` | Visual validation |
| 6. Ridge Regression | Implementation + Notebook | `src/ridge_regression_gd.py` + notebook | pytest + visual |

**Updated Project Structure:**
```
project/
├── src/
│   ├── __init__.py
│   ├── linear_regression_gd.py          # Main exercise
│   ├── linear_regression_sgd.py         # Exercise 1
│   ├── linear_regression_minibatch_gd.py # Exercise 2
│   └── ridge_regression_gd.py           # Exercise 6
├── notebooks/
│   ├── learning_rate_finder.ipynb       # Exercise 3
│   ├── momentum.ipynb                   # Exercise 4
│   ├── early_stopping.ipynb             # Exercise 5
│   └── ridge_regression.ipynb           # Exercise 6 (analysis)
├── tests/
│   ├── test_linear_regression_gd.py     # Tests for main + Ex 1-2
│   ├── test_gradient_descent_simple.py  # Simplified tests
│   └── test_ridge_regression_gd.py      # Tests for Exercise 6
└── README.md
```

## When to Use Gradient Descent vs. Closed-Form Solution

Let's summarize when each approach is most appropriate:

### Use Closed-Form Solution When:
- Number of features $d < 10,000$
- The matrix $\mathbf{X}^T\mathbf{X}$ is invertible
- You need the exact optimal solution
- The entire dataset fits in memory
- You're doing batch learning (not online)

### Use Gradient Descent When:
- Number of features $d > 10,000$
- Dataset is too large to fit in memory
- You need online learning capabilities
- $\mathbf{X}^T\mathbf{X}$ is singular or nearly singular
- You're using regularization (e.g., L1, L2)
- You want to extend to non-linear models (neural networks)

## Computational Complexity Comparison

| Operation | Closed-Form | Batch GD (per iteration) | SGD (per iteration) | Mini-Batch GD (per iteration) |
|-----------|-------------|--------------------------|---------------------|-------------------------------|
| Time Complexity | $O(d^3 + d^2n)$ | $O(dn)$ | $O(d)$ | $O(db)$ |
| Space Complexity | $O(d^2)$ | $O(d)$ | $O(d)$ | $O(d)$ |

where:
- $n$ = number of samples
- $d$ = number of features
- $b$ = mini-batch size

For large $d$, gradient descent has a significant computational advantage!

## Practical Tips and Best Practices

### 1. **Always Scale Features**
Standardization or min-max scaling dramatically improves convergence.

### 2. **Monitor the Loss**
Always plot the loss function to verify convergence. If the loss increases, your learning rate is too high.

### 3. **Start with a Learning Rate Grid Search**
Try learning rates: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] and select the best.

### 4. **Use Mini-Batch GD as Default**
For most problems, mini-batch gradient descent (batch size 32-256) works best.

### 5. **Check Your Gradient Implementation**
Use numerical gradient checking to verify your gradient calculations:

```python
def numerical_gradient(theta, X, y, epsilon=1e-5):
    """Compute gradient numerically for debugging"""
    grad = np.zeros_like(theta)
    
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        
        loss_plus = compute_loss(X, y, theta_plus)
        loss_minus = compute_loss(X, y, theta_minus)
        
        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return grad

# Compare analytical and numerical gradients
analytical_grad = compute_gradient(X, y, theta)
numerical_grad = numerical_gradient(theta, X, y)
difference = np.linalg.norm(analytical_grad - numerical_grad)
print(f"Gradient difference: {difference}")  # Should be < 1e-7
```

## Wrapping Up

In this tutorial, we've explored gradient descent as a powerful alternative to the closed-form solution for linear regression. Key takeaways include:

1. **Gradient descent** is an iterative optimization algorithm that finds the minimum of a loss function by taking steps proportional to the negative gradient.

2. **The learning rate** $\alpha$ is a critical hyperparameter that controls step size—too small leads to slow convergence, too large leads to divergence.

3. **Feature scaling** is essential for fast convergence—always standardize your features!

4. **When to use gradient descent**:
   - Large number of features ($d > 10,000$)
   - Large datasets that don't fit in memory
   - Online learning scenarios
   - When extending to regularized or non-linear models

5. **Gradient descent enables modern machine learning**, as it's the foundation for training neural networks and other complex models.

While the closed-form solution provides an exact answer in one computation, gradient descent offers flexibility, scalability, and serves as a gateway to understanding how deep learning models are trained!

## Further Reading

For those interested in diving deeper:

1. **Convex Optimization** by Boyd & Vandenberghe—comprehensive treatment of optimization theory
2. **Advanced gradient descent variants**: Momentum, RMSprop, Adam
3. **Second-order methods**: Newton's method, L-BFGS
4. **Coordinate descent** for high-dimensional problems
5. **Stochastic optimization theory** and convergence guarantees

---

*This tutorial is part of an ongoing series on machine learning fundamentals.*