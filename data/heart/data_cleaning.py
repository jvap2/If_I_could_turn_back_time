import polars as pl
import numpy as np

# Load the data
df = pl.read_csv("heart-disease.csv")

# Look at the columns and change the data
# The columns with words instead of values include: Cloud Cover, Season, Location, and Weather Type

# Change the Cloud Cover column

df_age = df["age"]
##find max and min age, then normalize the age column
print(df_age.max(), df_age.min())
df = df.with_columns((df["age"] - df_age.min()) / (df_age.max() - df_age.min()))



df_trestbps = df["trestbps"]
##find max and min trestbps, then normalize the trestbps column
print(df_trestbps.max(), df_trestbps.min())
df = df.with_columns((df["trestbps"] - df_trestbps.min()) / (df_trestbps.max() - df_trestbps.min()))

df_chol = df["chol"]
##find max and min chol, then normalize the chol column
print(df_chol.max(), df_chol.min())
df = df.with_columns((df["chol"] - df_chol.min()) / (df_chol.max() - df_chol.min()))

df_thalach = df["thalach"]
##find max and min thalach, then normalize the thalach column
print(df_thalach.max(), df_thalach.min())
df = df.with_columns((df["thalach"] - df_thalach.min()) / (df_thalach.max() - df_thalach.min()))

df_oldpeak = df["oldpeak"]
##find max and min oldpeak, then normalize the oldpeak column
print(df_oldpeak.max(), df_oldpeak.min())
# Save the cleaned data

# Shuffle the data
df = df[np.random.permutation(len(df))]

df.write_csv("heart_classification_data_cleaned.csv")