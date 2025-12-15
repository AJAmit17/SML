import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/Housing.csv")

# Scatter plot: Price vs Area
plt.scatter(df["area"], df["price"], alpha=0.6)
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Price vs Area")
plt.show()

# Boxplot: Price vs Bedrooms
plt.figure(figsize=(6,4))
sns.boxplot(x="bedrooms", y="price", data=df)
plt.title("Price vs Number of Bedrooms")
plt.show()


# Boxplot: Price vs Furnishing Status
plt.figure(figsize=(6,4))
sns.boxplot(x="furnishingstatus", y="price", data=df)
plt.title("Price vs Furnishing Status")
plt.show()

# Boxplot: Price vs Air Conditioning
plt.figure(figsize=(6,4))
sns.boxplot(x="airconditioning", y="price", data=df)
plt.title("Price vs Air Conditioning")
plt.show()
