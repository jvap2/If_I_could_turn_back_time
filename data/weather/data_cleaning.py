import polars as pl

# Load the data
df = pl.read_csv("weather_classification_data.csv")

# Look at the columns and change the data
# The columns with words instead of values include: Cloud Cover, Season, Location, and Weather Type

# Change the Cloud Cover column

df_cloud = df["Cloud Cover"]
print(df_cloud.value_counts())
cloud_dict = {"cloudy": 1, "partly cloudy": 2, "clear": 3, "overcast": 4}
df = df.with_columns(df["Cloud Cover"].map_dict(cloud_dict))
print(df["Cloud Cover"].value_counts())

# Change the Season column
df_season = df["Season"]
print(df_season.value_counts())
season_dict = {"winter": 1, "spring": 2, "summer": 3, "fall": 4}
df = df.with_columns(df["Season"].map_dict(season_dict))
print(df["Season"].value_counts())

# Change the Location column
df_location = df["Location"]
print(df_location.value_counts())
location_dict = {"inland": 1, "coastal": 2, "mountain": 3}
df = df.with_columns(df["Location"].map_dict(location_dict))

# Change the Weather Type column
df_weather = df["Weather Type"]
print(df_weather.value_counts())
weather_dict = {"Cloudy": 1, "Sunny": 2, "Rainy": 3, "Snowy": 4}
df = df.with_columns(df["Weather Type"].map_dict(weather_dict))

# Save the cleaned data
df.write_csv("weather_classification_data_cleaned.csv")
