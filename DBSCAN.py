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


from flask import Flask, request, render_template
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is uploaded successfully
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Read the uploaded CSV file
        df = pd.read_csv(file)
        
        # Create approximate locations using 'WILAYAH', 'KECAMATAN', and 'NAMA FASKES'
        df['location'] = df['WILAYAH'] + ', ' + df['KECAMATAN'] + ', ' + df['NAMA FASKES (Rumah Sakit dan Puskesmas)']
        
        # Drop duplicate locations to avoid redundancy
        df = df.drop_duplicates(subset=['location'])
        
        # Geocode the approximate locations
        geolocator = Nominatim(user_agent="geoapiExercises")
        df['location'] = df['location'].apply(lambda x: geolocator.geocode(x))
        
        # Drop rows with missing geocoding results
        df = df.dropna(subset=['location'])
        
        # Extract latitude and longitude from geocoding results
        df['latitude'] = df['location'].apply(lambda x: x.latitude)
        df['longitude'] = df['location'].apply(lambda x: x.longitude)
        
        # DBSCAN clustering
        X = df[['latitude', 'longitude']].values
        dbscan = DBSCAN(eps=0.1, min_samples=10)
        labels = dbscan.fit_predict(X)
        
        # Visualization
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        
        # Convert plot to base64 image
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Render template with plot
        return render_template('result.html', plot_url=plot_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
