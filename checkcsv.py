import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from math import cos, sin, radians
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load data from CSV
data = pd.read_csv('new_januari.csv')

# Drop rows with missing values
data.dropna(subset=['latitude', 'longitude', 'Jenis Penyakit'], inplace=True)

# Encode disease types into numerical values
le = LabelEncoder()
data['disease_encoded'] = le.fit_transform(data['Jenis Penyakit'])

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

# Convert latitude and longitude to Cartesian coordinates
def lat_lon_to_cartesian(lat, lon):
    R = 6371000  # Earth radius in meters
    lat_rad = radians(lat)
    lon_rad = radians(lon)
    x = R * cos(lat_rad) * cos(lon_rad)
    y = R * cos(lat_rad) * sin(lon_rad)
    return x, y

gdf['x'], gdf['y'] = zip(*gdf.apply(lambda row: lat_lon_to_cartesian(row['latitude'], row['longitude']), axis=1))

# Define the number of clusters for K-Means
n_clusters = 10  # Adjust as needed

# Perform K-Means clustering with both coordinates and disease types
kmeans = KMeans(n_clusters=n_clusters)
gdf['cluster'] = kmeans.fit_predict(gdf[['x', 'y', 'disease_encoded']])

# Create a map using Folium
map_clusters = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=10)

# Define colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange']#, 'pink', 'cyan', 'yellow', 'gray', 'teal']

# Add markers for each data point
for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=colors[row['cluster'] % len(colors)],
        fill=True,
        fill_color=colors[row['cluster'] % len(colors)]
    ).add_to(map_clusters)

# Save and display the map
map_clusters.save('penyakitkmean.html')
map_clusters  # This will display the map in Jupyter notebook environments
