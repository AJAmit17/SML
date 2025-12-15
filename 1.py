import numpy as np
import matplotlib.pyplot as plt

bills = [850, 900, 950, 980, 1000, 1020, 1050, 1075, 1100, 1125,
         1150, 1175, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
         1600, 1700, 1800, 1900, 2100, 2300]

Q1 = np.percentile(bills, 25)
Q3 = np.percentile(bills, 75)
IQR = Q3 - Q1

print("Q1 (25th percentile):", Q1)
print("Q3 (75th percentile):", Q3)
print("Interquartile Range (IQR):", IQR)

plt.boxplot(bills)
plt.title("Electricity Bills - Boxplot Representation")
plt.xlabel("Bill Amount (₹)")
plt.show()
