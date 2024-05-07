import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Load data from CSV file
data = pd.read_csv('januari.csv', delimiter=';', skiprows=4)

# Assuming your CSV file has latitude and longitude columns, you can select those columns
X = data[['latitude', 'longitude']].values

# Agglomerative hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clustering.fit(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 1], X[:, 0], c=agg_clustering.labels_, cmap='viridis', s=50, alpha=0.7) # Swap longitude and latitude for plotting
plt.title('Agglomerative Hierarchical Clustering in Surabaya')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
