import numpy as np
from sklearn import cluster
from Cluster import createClusters, Cluster
from Behavior import makePointList, Behavior
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def best_k(points):
  k_list = []

  for k in range(2, 11):
    kmeans = KMeans(n_clusters = k).fit(points)
    labels = kmeans.labels_
    k_list.append(silhouette_score(points, labels, metric = 'euclidean'))
    
  return k_list.index(max(k_list)) + 2

def findCentroids(points, k):
  kmeans = cluster.KMeans(n_clusters=k)
  kmeans.fit(points)
  centroids = kmeans.cluster_centers_
  return centroids

