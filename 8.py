import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and prepare data
data = pd.read_csv('datasets/titanic.csv')
data['sex'] = data['sex'].map({'male':0, 'female':1})
data['age'] = data['age'].fillna(data['age'].median())
X = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
y = data['survived']

# Step 2: Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Linear SVM
model1 = SVC(kernel='linear', random_state=42)
model1.fit(X_train, y_train)
print("Linear SVM Accuracy:", accuracy_score(y_test, model1.predict(X_test)))
print(classification_report(y_test, model1.predict(X_test)))

# Step 4: RBF SVM
model2 = SVC(kernel='rbf', random_state=42)
model2.fit(X_train, y_train)
print("\nRBF SVM Accuracy:", accuracy_score(y_test, model2.predict(X_test)))
print(classification_report(y_test, model2.predict(X_test)))
