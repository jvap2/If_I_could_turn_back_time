import polars as pl
import numpy as np
import matplotlib.pyplot as plt

'''
We need to properly analyze the data and scale the data to make it easier to work with.
Linear Scaling:
    x = (x - min(x)) / (max(x) - min(x))

Z-Score Scaling:
    x = (x - mean(x)) / std(x)

Log Scaling:
    x = log(x)

Linear Scaling shoudl be used when the features are uniformly distributed across a fixed range

Z-Score Scaling should be used when the features are normally distributed with no extreme outliers

Log Scaling should be used when the features are skewed, i.e. comforms to power law distribution
'''

df = pl.read_csv("heart-disease.csv")

df_age = df["age"]

# Look at the statistics of the age column
print(df_age.describe())
'''
│ str        ┆ f64       │
╞════════════╪═══════════╡
│ count      ┆ 303.0     │
│ null_count ┆ 0.0       │
│ mean       ┆ 54.366337 │
│ std        ┆ 9.082101  │
│ min        ┆ 29.0      │
│ max        ┆ 77.0      │
│ median     ┆ 55.0      │
│ 25%        ┆ 47.0      │
│ 75%        ┆ 61.0      │

'''

## Plot the age column
plt.hist(df_age.to_numpy(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_distribution.png")
plt.clf()

''' Figure out how to alter the data to make it more normally distributed '''

df_trestbps = df["trestbps"]

# Look at the statistics of the trestbps column
print(df_trestbps.describe())


## Plot the trestbps column
plt.hist(df_trestbps.to_numpy(), bins=20)

plt.title("Trestbps Distribution")
plt.xlabel("Trestbps")
plt.ylabel("Frequency")
plt.savefig("trestbps_distribution.png")
plt.clf()

''' Figure out how to alter the data to make it more normally distributed '''

df_chol = df["chol"]

# Look at the statistics of the chol column
print(df_chol.describe())

## Plot the chol column
plt.hist(df_chol.to_numpy(), bins=20)
plt.title("Chol Distribution")
plt.xlabel("Chol")
plt.ylabel("Frequency")
plt.savefig("chol_distribution.png")
plt.clf()

''' Figure out how to alter the data to make it more normally distributed '''

df_thalach = df["thalach"]

# Look at the statistics of the thalach column
print(df_thalach.describe())

## Plot the thalach column
plt.hist(df_thalach.to_numpy(), bins=20)
plt.title("Thalach Distribution")
plt.xlabel("Thalach")
plt.ylabel("Frequency")
plt.savefig("thalach_distribution.png")
plt.clf()

''' Figure out how to alter the data to make it more normally distributed '''

df_oldpeak = df["oldpeak"]

# Look at the statistics of the oldpeak column
print(df_oldpeak.describe())

## Plot the oldpeak column
plt.hist(df_oldpeak.to_numpy(), bins=20)
plt.title("Oldpeak Distribution")
plt.xlabel("Oldpeak")
plt.ylabel("Frequency")
plt.savefig("oldpeak_distribution.png")
plt.clf()

''' Figure out how to alter the data to make it more normally distributed '''


