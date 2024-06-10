import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from math import cos, sin, radians
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Read the CSV file
data = pd.read_csv('new_februari.csv')

# Drop rows with missing latitude, longitude, or disease type
data.dropna(subset=['latitude', 'longitude', 'Jenis Penyakit'], inplace=True)

# Encode disease types into numerical values
le = LabelEncoder()
data['datapenyakit'] = le.fit_transform(data['Jenis Penyakit'])

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

# Prepare data for KMeans
X = gdf[['x', 'y', 'datapenyakit']]

# Standardize features for better performance of KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform KMeans clustering
n_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
gdf['cluster'] = kmeans.fit_predict(X_scaled)

# Jitter function to avoid overlapping
def jitter(lat, lon, jitter_amount=0.0002):
    jittered_lat = lat + np.random.uniform(-jitter_amount, jitter_amount)
    jittered_lon = lon + np.random.uniform(-jitter_amount, jitter_amount)
    return jittered_lat, jittered_lon

# Create a color map for diseases
unique_diseases = gdf['Jenis Penyakit'].unique()
colors = plt.get_cmap('tab20')  # Use the updated way to get the colormap
color_list = colors(np.linspace(0, 1, len(unique_diseases)))

# Create a mapping from disease to color
disease_color_map = {disease: color for disease, color in zip(unique_diseases, color_list)}

# Convert RGBA to hex
disease_color_map_hex = {disease: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for disease, (r, g, b, a) in disease_color_map.items()}

# Create a map using Folium
map_clusters = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=10)

# Add markers for each data point
for _, row in gdf.iterrows():
    jittered_lat, jittered_lon = jitter(row['latitude'], row['longitude'])
    color = disease_color_map_hex[row['Jenis Penyakit']]
    popup_text = f"Disease: {row['Jenis Penyakit']}"
    folium.CircleMarker(
        location=[jittered_lat, jittered_lon],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        popup=popup_text
    ).add_to(map_clusters)

map_clusters.save('KMEANSFEB.html')

# To display the map in a Jupyter notebook, use:
# map_clusters
