import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 1: Create simple dataset
np.random.seed(42)
X1 = np.random.normal(0, 1, 500)
X2 = 0.5*X1 + np.random.normal(0, 1, 500)
y = np.random.choice([0, 1], size=500, p=[0.9, 0.1])
df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

# Step 2: Show covariance matrix
print("Covariance Matrix:\n", df[['X1', 'X2']].cov())

# Step 3: Apply LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(df[['X1','X2']], df['y'])
plt.scatter(X_lda[df['y']==0], np.zeros(sum(df['y']==0)), label='Class 0')
plt.scatter(X_lda[df['y']==1], np.zeros(sum(df['y']==1)), label='Class 1')
plt.title("LDA Projection")
plt.legend()
plt.show()

# Step 4: Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(df[['X1','X2']], df['y'], test_size=0.2, random_state=42)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

# Step 5: Show odds ratios
coef = model.coef_[0]
for i, name in enumerate(['X1','X2']):
    print(f"{name}: Coef={coef[i]:.2f}, Odds Ratio={np.exp(coef[i]):.2f}")
