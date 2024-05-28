import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from math import cos, sin, radians
import folium
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('new_januari.csv')

# Drop rows with missing values
data.dropna(subset=['latitude', 'longitude'], inplace=True)

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
n_clusters = 3  # Adjust as needed

# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_clusters)
gdf['cluster'] = kmeans.fit_predict(gdf[['x', 'y']])

# Plot the clusters using Matplotlib
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, column='cluster', legend=True, cmap='tab20', markersize=5)
plt.title('K-Means Clustering of Geospatial Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Create a map using Folium
map_clusters = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=10)

# Add markers for each data point
for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='blue' if row['cluster'] == -1 else 'red',
        fill=True,
        fill_color='blue' if row['cluster'] == -1 else 'red'
    ).add_to(map_clusters)

# Save and display the map
map_clusters.save('kmean.html')
map_clusters  # This will display the map in Jupyter notebook environments

##PAKE PLOTLY
# import plotly.express as px

# # Load data from CSV
# data = pd.read_csv('new_januari.csv')

# # Drop rows with missing values
# data.dropna(subset=['latitude', 'longitude'], inplace=True)

# # Create a GeoDataFrame
# gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

# # Convert latitude and longitude to Cartesian coordinates
# def lat_lon_to_cartesian(lat, lon):
#     R = 6371000  # Earth radius in meters
#     lat_rad = radians(lat)
#     lon_rad = radians(lon)
#     x = R * cos(lat_rad) * cos(lon_rad)
#     y = R * cos(lat_rad) * sin(lon_rad)
#     return x, y

# gdf['x'], gdf['y'] = zip(*gdf.apply(lambda row: lat_lon_to_cartesian(row['latitude'], row['longitude']), axis=1))

# # Define the number of clusters for K-Means
# n_clusters = 3  # Adjust as needed

# # Perform K-Means clustering
# kmeans = KMeans(n_clusters=n_clusters)
# gdf['cluster'] = kmeans.fit_predict(gdf[['x', 'y']])

# # Create an interactive map using Plotly
# fig = px.scatter_mapbox(gdf, lat='latitude', lon='longitude', color='cluster',
#                         mapbox_style='carto-positron', zoom=10, height=600, width=800)

# fig.update_layout(mapbox=dict(center=dict(lat=gdf['latitude'].mean(), lon=gdf['longitude'].mean())))
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()
