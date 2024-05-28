# import pandas as pd
# from sklearn.cluster import DBSCAN
# from sklearn.metrics.pairwise import haversine_distances
# from math import radians
# import folium

# # Load data from CSV
# data = pd.read_csv('new_januari.csv')

# # Drop rows with missing values
# data.dropna(subset=['latitude', 'longitude'], inplace=True)

# # Convert latitude and longitude to radians
# data['latitude'] = data['latitude'].apply(radians)
# data['longitude'] = data['longitude'].apply(radians)

# # Define epsilon and minimum samples for DBSCAN
# epsilon = 1.5  # Adjust as needed
# min_samples = 2  # Adjust as needed

# # Define haversine function to calculate distance
# def haversine(point1, point2):
#     return haversine_distances([point1, point2])[0][1] * 6371000  # Earth radius in meters

# # Define a function to perform DBSCAN clustering
# def dbscan_clustering(data, epsilon, min_samples):
#     dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric=haversine)
#     clusters = dbscan.fit_predict(data[['latitude', 'longitude']])
#     return clusters

# # Perform clustering
# data['cluster'] = dbscan_clustering(data, epsilon, min_samples)

# # Create a map
# map_clusters = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=10)

# # Add markers for each data point
# for index, row in data.iterrows():
#     folium.CircleMarker(
#         location=[row['latitude'], row['longitude']],
#         radius=5,
#         color='blue' if row['cluster'] == -1 else 'red',
#         fill=True,
#         fill_color='blue' if row['cluster'] == -1 else 'red'
#     ).add_to(map_clusters)

# # Display the map
# map_clusters.save('map.html')  # Save map as HTML

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import folium
import matplotlib.pyplot as plt

data = pd.read_csv('new_januari.csv')

data.dropna(subset=['latitude', 'longitude'], inplace=True)

gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

gdf['latitude_rad'] = gdf['latitude'].apply(radians)
gdf['longitude_rad'] = gdf['longitude'].apply(radians)


epsilon = 2 / 6371000  # Convert meters to radians by dividing by Earth radius
min_samples = 1  # Adjust as needed

# Define haversine function to calculate distance
def haversine(point1, point2):
    return haversine_distances([point1, point2])[0][1]  # Result is in radians

# Define a function to perform DBSCAN clustering
def dbscan_clustering(data, epsilon, min_samples):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric=haversine)
    clusters = dbscan.fit_predict(data[['latitude_rad', 'longitude_rad']])
    return clusters

# Perform clustering
gdf['cluster'] = dbscan_clustering(gdf, epsilon, min_samples)

# Plot the clusters using Matplotlib
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, column='cluster', legend=True, cmap='tab20', markersize=5)
plt.title('DBSCAN Clustering of Geospatial Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Create a map using Folium
map_clusters = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=10)

for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='blue' if row['cluster'] == -1 else 'red',
        fill=True,
        fill_color='blue' if row['cluster'] == -1 else 'red'
    ).add_to(map_clusters)

map_clusters.save('map.html')
# map_clusters
