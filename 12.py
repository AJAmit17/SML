import numpy as np
import pandas as pd

# Simulate dataset
np.random.seed(42)
n_samples = 500
X1 = np.random.normal(0, 1, n_samples)
X2 = 0.5*X1 + np.random.normal(0, 1, n_samples)
y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # Imbalanced

# Create DataFrame
df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

# Covariance matrix of input features
cov_matrix = df[['X1', 'X2']].cov()
print("Covariance Matrix:\n", cov_matrix)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(df[['X1','X2']], df['y'])

# Plot projected data
plt.scatter(X_lda[df['y']==0], np.zeros(sum(df['y']==0)), label='Class 0', alpha=0.5)
plt.scatter(X_lda[df['y']==1], np.zeros(sum(df['y']==1)), label='Class 1', alpha=0.5)
plt.title("LDA Projection")
plt.legend()
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df[['X1','X2']], df['y'], test_size=0.2, random_state=42)

# Logistic Regression
logreg = LogisticRegression(class_weight='balanced', random_state=42)
logreg.fit(X_train, y_train)

# Predict
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))

# Coefficients
coefficients = logreg.coef_[0]
odds_ratios = np.exp(coefficients)

for i, feature in enumerate(['X1','X2']):
    print(f"Feature: {feature}, Coefficient: {coefficients[i]:.3f}, Odds Ratio: {odds_ratios[i]:.3f}")
