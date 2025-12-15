import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Step 1: Generate sample nonlinear data
np.random.seed(42)
sqft = np.linspace(800, 3500, 100)
# Nonlinear price behavior (change in slope after 2000 sqft)
price = 150 * sqft + np.where(sqft > 2000, 100 * (sqft - 2000), 0) + np.random.normal(0, 50000, 100)

# Step 2: Create DataFrame
data = pd.DataFrame({'sqft': sqft, 'price': price})

# Step 3: Define spline feature with a knot at 2000 sqft
data['spline_term'] = np.where(data['sqft'] > 2000, data['sqft'] - 2000, 0)

# Step 4: Fit spline regression model
X = sm.add_constant(data[['sqft', 'spline_term']])
model = sm.OLS(data['price'], X).fit()

# Step 5: Display model summary
print(model.summary())

# Step 6: Predict and visualize
data['pred_price'] = model.predict(X)

plt.figure(figsize=(8,5))
plt.scatter(data['sqft'], data['price'], color='gray', label='Actual data')
plt.plot(data['sqft'], data['pred_price'], color='red', linewidth=2, label='Spline Regression Fit')
plt.axvline(2000, color='blue', linestyle='--', label='Knot at 2000 sqft')
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.title('Spline Regression Model (Knot at 2000 sqft)')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Interpretation
print("\nInterpretation:")
print("β₁ (sqft):", round(model.params['sqft'], 2), "→ Slope before 2000 sqft")
print("β₂ (spline_term):", round(model.params['spline_term'], 2), "→ Change in slope after 2000 sqft")
print("New slope after 2000 sqft =", round(model.params['sqft'] + model.params['spline_term'], 2))
