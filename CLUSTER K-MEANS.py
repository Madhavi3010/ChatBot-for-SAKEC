import pandas as pd
from sklearn.cluster import KMeans

# Load the Iris dataset from CSV
iris_data = pd.read_csv('iris.csv')

# Extract the features from the dataset
X = iris_data.iloc[:, :-1].values

# Create a K-means clustering object with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the data to the K-means object
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Print the cluster labels and centroids
print("Cluster Labels:", labels)
print("Cluster Centroids:", centroids)
