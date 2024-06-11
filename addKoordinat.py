import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

try:
    df = pd.read_csv("februari.csv", sep=';')
except pd.errors.ParserError as e:
    print(f"Error reading CSV: {e}")

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

df.to_csv("new_februari.csv", index=False)




#standarisasi nama lokasi supaya semua nama tempatna bisa ketemu
# import pandas as pd
# from geopy.geocoders import Nominatim
# import time

# # Initialize the geolocator
# geolocator = Nominatim(user_agent="geoapiExercises")

# # Read the CSV file
# try:
#     df = pd.read_csv("februari.csv", sep=';')
# except pd.errors.ParserError as e:
#     print(f"Error reading CSV: {e}")

# # Function to clean and standardize addresses
# def clean_address(address):
#     address = address.strip().lower()
#     return address

# # Function to geocode an address
# def geocode_address(row):
#     address = clean_address(row["NAMA FASKES (Rumah Sakit dan Puskesmas)"])
#     location = geolocator.geocode(address)
#     if location:
#         row["latitude"] = location.latitude
#         row["longitude"] = location.longitude
#     else:
#         # Retry with a modified address (e.g., removing details, using city and country)
#         location = geolocator.geocode(address.split(",")[0])  # Trying with just the first part of the address
#         if location:
#             row["latitude"] = location.latitude
#             row["longitude"] = location.longitude
#         else:
#             row["latitude"] = None
#             row["longitude"] = None
#     return row

# # Apply the geocoding function with a delay to avoid hitting the rate limit
# df = df.apply(lambda row: geocode_address(row) if pd.isnull(row.get("latitude")) or pd.isnull(row.get("longitude")) else row, axis=1)

# # Save the new DataFrame to a CSV file
# df.to_csv("februari_try.csv", index=False)
