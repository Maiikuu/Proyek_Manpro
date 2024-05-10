# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import AgglomerativeClustering

# # Load data from CSV file
# data = pd.read_csv('new_januari.csv', delimiter=';')

# # Assuming your CSV file has latitude and longitude columns, you can select those columns
# X = data[['latitude', 'longitude']].values

# # Agglomerative hierarchical clustering
# agg_clustering = AgglomerativeClustering(n_clusters=3)
# agg_clustering.fit(X)

# # Plotting the clusters
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 1], X[:, 0], c=agg_clustering.labels_, cmap='viridis', s=50, alpha=0.7) # Swap longitude and latitude for plotting
# plt.title('Agglomerative Hierarchical Clustering in Surabaya')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()


import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import matplotlib.pyplot as plt

data = pd.read_csv('new_januari.csv')

# Drop rows with missing values
data.dropna(subset=['latitude', 'longitude'], inplace=True)

# Convert latitude and longitude to radians
data['latitude'] = data['latitude'].apply(radians)
data['longitude'] = data['longitude'].apply(radians)

# Define epsilon and minimum samples for DBSCAN
epsilon = 1.5  # Adjust as needed
min_samples = 2  # Adjust as needed

# Define haversine function to calculate distance
def haversine(point1, point2):
    return haversine_distances([point1, point2])[0][1] * 6371000  # Earth radius in meters

# Define a function to perform DBSCAN clustering
def dbscan_clustering(data, epsilon, min_samples):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric=haversine)
    clusters = dbscan.fit_predict(data[['latitude', 'longitude']])
    return clusters

# Perform clustering
data['cluster'] = dbscan_clustering(data, epsilon, min_samples)

# Plot clusters
plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='viridis', s=20)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
