import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(0)
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = 2 * X**2 + 3 * X + 5 + np.random.randn(30, 1) * 2

# Try different degrees
degrees = [1, 2, 3, 4, 5]
plt.figure(figsize=(12, 8))

for i, d in enumerate(degrees, 1):
    # Transform and fit
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict and calculate error
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    
    # Plot
    plt.subplot(2, 3, i)
    plt.scatter(X, y, color='black')
    plt.plot(X, y_pred, color='red')
    plt.title(f'Degree {d} | MSE: {mse:.2f}')

plt.tight_layout()
plt.show()