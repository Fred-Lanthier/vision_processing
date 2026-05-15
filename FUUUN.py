import numpy as np
from viking_kalman import StateSpaceModel

# 1. Simulate data
np.random.seed(1)
n, d = 100, 5
Q = np.diag([0, 0, 0.25, 0.25, 0.25]) # Process noise covariance
sig = 1 # Measurement noise standard deviation
X = np.hstack([np.random.randn(n, d - 1), np.ones((n, 1))])

theta = np.random.randn(d, 1)
theta_arr = np.zeros((n, d))
for t in range(n):
    theta_arr[t, :] = theta.flatten()
    theta = theta + np.random.multivariate_normal(np.zeros(d), Q).reshape((d, 1))
    
y = np.sum(X * theta_arr, axis=1) + np.random.normal(0, sig, size=n)

# 2. Build the model and visualize diagnostics
ssm = StateSpaceModel(X, y, kalman_params={'Q': Q, 'sig': sig})
ssm.plot()