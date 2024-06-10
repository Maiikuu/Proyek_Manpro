import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from math import cos, sin, radians
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('new_januari.csv')

data.dropna(subset=['latitude', 'longitude', 'Jenis Penyakit'], inplace=True)

le = LabelEncoder()
data['datapenyakit'] = le.fit_transform(data['Jenis Penyakit'])

gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

def lat_lon_to_cartesian(lat, lon):
    R = 6371000  # Earth radius in meters
    lat_rad = radians(lat)
    lon_rad = radians(lon)
    x = R * cos(lat_rad) * cos(lon_rad)
    y = R * cos(lat_rad) * sin(lon_rad)
    return x, y

gdf['x'], gdf['y'] = zip(*gdf.apply(lambda row: lat_lon_to_cartesian(row['latitude'], row['longitude']), axis=1))

n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters)
gdf['cluster'] = kmeans.fit_predict(gdf[['x', 'y', 'datapenyakit']])

map_clusters = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=10)

colors = ['red', 'blue', 'green', 'purple', 'orange']#, 'pink', 'cyan', 'yellow', 'gray', 'teal']

for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=colors[row['cluster'] % len(colors)],
        fill=False,
        fill_color=colors[row['cluster'] % len(colors)]
    ).add_to(map_clusters)

map_clusters.save('new_februari.html')
# map_clusters  #display map di jupyter notebook
