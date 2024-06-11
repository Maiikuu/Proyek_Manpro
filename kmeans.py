import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from math import cos, sin, radians
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

data = pd.read_csv('new_februari.csv')

data.dropna(subset=['latitude', 'longitude', 'Jenis Penyakit'], inplace=True)

le = LabelEncoder()
data['datapenyakit'] = le.fit_transform(data['Jenis Penyakit'])

gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

def lat_lon_to_cartesian(lat, lon):
    R = 6371000
    lat_rad = radians(lat)
    lon_rad = radians(lon)
    x = R * cos(lat_rad) * cos(lon_rad)
    y = R * cos(lat_rad) * sin(lon_rad)
    return x, y

gdf['x'], gdf['y'] = zip(*gdf.apply(lambda row: lat_lon_to_cartesian(row['latitude'], row['longitude']), axis=1))

X = gdf[['x', 'y', 'datapenyakit']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 5  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
gdf['cluster'] = kmeans.fit_predict(X_scaled)


def jitter(lat, lon, jitter_amount=0.0002):
    jittered_lat = lat + np.random.uniform(-jitter_amount, jitter_amount)
    jittered_lon = lon + np.random.uniform(-jitter_amount, jitter_amount)
    return jittered_lat, jittered_lon

unique_diseases = gdf['Jenis Penyakit'].unique()
colors = plt.get_cmap('tab20_r')
color_list = colors(np.linspace(0, 1, len(unique_diseases)))

disease_color_map = {disease: color for disease, color in zip(unique_diseases, color_list)}

disease_color_map_hex = {disease: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for disease, (r, g, b, a) in disease_color_map.items()}

map_clusters = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=10)

for _, row in gdf.iterrows():
    jittered_lat, jittered_lon = jitter(row['latitude'], row['longitude'])
    color = disease_color_map_hex[row['Jenis Penyakit']]
    popup_text = f"{row['Jenis Penyakit']}"
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
