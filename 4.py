import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define a fixed right-skewed population (100 users for simplicity)
population = [
    3000, 3500, 4000, 4200, 4300, 4500, 4600, 4700, 4800, 5000,
    5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
    6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100,
    7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100,
    8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100,
    9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10500,
    11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 
    20000, 22000, 25000, 27000, 30000, 35000, 40000, 45000, 50000, 60000,
    70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000,
    160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000
]

population = np.array(population)

# Step 2: Define 10 fixed samples (each of size 50)
samples = [
    population[0:50],
    population[10:60],
    population[20:70],
    population[30:80],
    population[40:90],
    population[50:100],
    np.concatenate([population[0:25], population[50:75]]),
    np.concatenate([population[10:35], population[60:85]]),
    np.concatenate([population[20:45], population[70:95]]),
    np.concatenate([population[5:30], population[80:105]])  # clipped at 100
]

# Step 3: Compute sample means
sample_means = [np.mean(s) for s in samples]

# Step 4: Plot histogram of sample means
plt.hist(sample_means, bins=5, edgecolor="black")
plt.xlabel("Sample Mean Step Count")
plt.ylabel("Frequency")
plt.title("Distribution of 10 Sample Means (n=50 each, fixed data)")
plt.show()

# Print results
print("Sample Means of 10 Samples:")
for i, mean in enumerate(sample_means, 1):
    print(f"Sample {i}: {mean:.2f}")

print("\nOverall Average of Sample Means:", np.mean(sample_means))
