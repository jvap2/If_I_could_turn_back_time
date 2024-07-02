import polars as pl

# Load the data
df = pl.read_csv("weather_classification_data.csv")

# Look at the columns and change the data
# The columns with words instead of values include: Cloud Cover, Season, Location, and Weather Type

# Change the Cloud Cover column

df_cloud = df["Cloud Cover"]
print(df_cloud.value_counts())
cloud_dict = {"cloudy": 0, "partly cloudy": 1, "clear": 2, "overcast": 3}
df = df.with_columns(df["Cloud Cover"].map_dict(cloud_dict))
print(df["Cloud Cover"].value_counts())

# Change the Season column
df_season = df["Season"]
print(df_season.value_counts())
season_dict = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
df = df.with_columns(df["Season"].map_dict(season_dict))
print(df["Season"].value_counts())

# Change the Location column
df_location = df["Location"]
print(df_location.value_counts())
location_dict = {"inland": 0, "coastal": 1, "mountain": 2}
df = df.with_columns(df["Location"].map_dict(location_dict))

# Change the Weather Type column
df_weather = df["Weather Type"]
print(df_weather.value_counts())
weather_dict = {"Cloudy": 0, "Sunny": 1, "Rainy": 2, "Snowy": 3}
df = df.with_columns(df["Weather Type"].map_dict(weather_dict))

# Save the cleaned data
df.write_csv("weather_classification_data_cleaned.csv")
