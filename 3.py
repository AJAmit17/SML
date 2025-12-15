import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example dataset
data = {
    'Battery_mAh': [3000, 4000, 4500, 5000, 6000, 3500, 4200],
    'Screen_inch': [5.5, 6.1, 6.4, 6.7, 6.8, 5.8, 6.2],
    'Price_Rs': [10000, 15000, 18000, 22000, 30000, 12000, 17000]
}

df = pd.DataFrame(data)

# Pair plot
sns.pairplot(df)
plt.show()

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
