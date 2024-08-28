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

''' Z-Score Scaling '''
df = df.with_columns((df["age"] - df["age"].mean()) / df["age"].std())

''' Plot the data after z-score scaling '''

plt.hist(df["age"].to_numpy(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_zscore_distribution.png")
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
''' Try log scaling'''

df = df.with_columns(np.log(df["trestbps"]))
print(df["trestbps"].describe())

''' Plot the data after log scaling '''

plt.hist(df["trestbps"].to_numpy(), bins=20)
plt.title("Trestbps Distribution")
plt.xlabel("Trestbps")
plt.ylabel("Frequency")
plt.savefig("trestbps_log_distribution.png")

plt.clf()



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
'''Z-score'''

df = df.with_columns((df["chol"] - df["chol"].mean()) / df["chol"].std())

''' Plot the data after z-score scaling '''

plt.hist(df["chol"].to_numpy(), bins=20)
plt.title("Chol Distribution")
plt.xlabel("Chol")
plt.ylabel("Frequency")
plt.savefig("chol_zscore_distribution.png")
plt.clf()


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
''' Use z score scaling '''

df = df.with_columns((df["thalach"] - df["thalach"].mean()) / df["thalach"].std())



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
''' This shows more of a power law distribution, so we should use log scaling '''

# df = df.with_columns((df["oldpeak"] - df["oldpeak"].mean()) / df["oldpeak"].std())

df = df.with_columns(np.log(np.log(df["oldpeak"]+1)+1))
print(df["oldpeak"].describe())
''' Plot the data after log scaling '''

plt.hist(df["oldpeak"].to_numpy(), bins=4)
plt.title("Oldpeak Distribution")
plt.xlabel("Oldpeak")
plt.ylabel("Frequency")
plt.savefig("oldpeak_log_distribution.png")
plt.clf()

df_cp = df["cp"]
print(df_cp.describe())

''' Plot the cp column '''
plt.hist(df_cp.to_numpy(), bins=4)
plt.title("CP Distribution")
plt.xlabel("CP")
plt.ylabel("Frequency")
plt.savefig("cp_distribution.png")
plt.clf()

''' Figure out how to alter the data to make it more normally distributed '''
''' Use z score scaling '''

''' Linear scaling'''

df = df.with_columns((df["cp"] - df["cp"].min()) / (df["cp"].max() - df["cp"].min()))

df_ca = df["ca"]
'''Linear scaling'''
df = df.with_columns((df["ca"] - df["ca"].min()) / (df["ca"].max() - df["ca"].min()))


df = df[np.random.permutation(len(df))]
# Save the cleaned data
df.write_csv("heart_classification_data_cleaned.csv")