import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Step 1: Generate data with knot at 2000
np.random.seed(42)
sqft = np.linspace(800, 3500, 100)
price = 150*sqft + np.where(sqft > 2000, 100*(sqft - 2000), 0) + np.random.normal(0, 50000, 100)
data = pd.DataFrame({'sqft': sqft, 'price': price})

# Step 2: Create spline term (knot at 2000)
data['spline'] = np.where(data['sqft'] > 2000, data['sqft'] - 2000, 0)

# Step 3: Fit model
X = sm.add_constant(data[['sqft', 'spline']])
model = sm.OLS(data['price'], X).fit()
print(model.summary())

# Step 4: Visualize
plt.scatter(data['sqft'], data['price'], color='gray', label='Actual')
plt.plot(data['sqft'], model.predict(X), color='red', linewidth=2, label='Spline Fit')
plt.axvline(2000, color='blue', linestyle='--', label='Knot')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Spline Regression')
plt.legend()
plt.show()

# Step 5: Interpretation
print(f"\nSlope before 2000: {model.params['sqft']:.2f}")
print(f"Change after 2000: {model.params['spline']:.2f}")
print(f"Slope after 2000: {model.params['sqft'] + model.params['spline']:.2f}")
