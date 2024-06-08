import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

# Read the CSV file, handle bad lines by skipping
try:
    df = pd.read_csv("februari.csv", sep=';')
except pd.errors.ParserError as e:
    print(f"Error reading CSV: {e}")

# Geocode the addresses and add latitude and longitude columns
def geocode_address(row):
    address = row["NAMA FASKES (Rumah Sakit dan Puskesmas)"]
    location = geolocator.geocode(address)
    if location:
        row["latitude"] = location.latitude
        row["longitude"] = location.longitude
    else:
        row["latitude"] = None
        row["longitude"] = None
    return row

df = df.apply(geocode_address, axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv("new_februari.csv", index=False)