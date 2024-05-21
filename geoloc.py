import pandas as pd
from geopy.geocoders import Nominatim

# Assuming your data is stored in a CSV file named 'healthcare_data.csv'
data = pd.read_csv('januari.csv')

# Filter the data for 'Puskesmas Asemrowo' in 'Surabaya Barat'
location_data = data[(data['WILAYAH'] == 'Surabaya Barat') & (data['NAMA FASKES (Rumah Sakit dan Puskesmas)'] == 'Puskesmas Asemrowo')]

# Get the unique address for the location
location_address = location_data['WILAYAH'].iloc[0] + ', ' + location_data['KECAMATAN'].iloc[0] + ', ' + location_data['NAMA_FASKES_(Rumah Sakit dan Puskesmas)'].iloc[0]

# Use geocoding to find the coordinates of the location
geolocator = Nominatim(user_agent="geoapiExercises")
location = geolocator.geocode(location_address)

if location:
    print("Latitude and Longitude of the location:")
    print((location.latitude, location.longitude))
else:
    print("Location not found.")

#masi error