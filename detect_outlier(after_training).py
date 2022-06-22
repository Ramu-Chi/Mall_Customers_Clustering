'''
NOTE: The detection results is not guaranteed outliers
Detect possible outliers after training with k = 5 (choosing_k_analyse.py):

Using the IQR method (only upper boundary) on the distances between each example and its cluster's centroid
-> We find the example whose distance is not ordinary (too big)
'''
import numpy as np
import pandas as pd

from my_model.distance_function import euclidean_distance
from my_model.k_mean import KMeans
from my_model.my_plot import plot_customer_clusters
from my_model.preprocessing import min_max_scaling, z_score_scaling

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# Scaling
X = z_score_scaling(customers)
X[:, 0] /= 3 # weight age = 1/3

# Training
km = KMeans(k=5)
km.fit(X)
labels = km.predict(X)

dists = []
for i in range(len(X)):
    centroid = km.centroid_list[labels[i]]
    dist = euclidean_distance(X[i], centroid)
    dists.append(dist)

Q1 = np.percentile(dists, 25)
Q3 = np.percentile(dists, 75)
IQR = Q3 - Q1

outliers = []
for i in range(len(X)):
    if dists[i] > Q3 + 1.5 * IQR:
        outliers.append(customers[i])
outliers = np.array(outliers)

print('Possible Outliers:')
for outlier in outliers:
    for i, cus in enumerate(customers):
        if (cus == outlier).all(): break
    print('customer id %d:' % (i + 1), outlier)

plot_customer_clusters(customers, labels, FIELD, outliers=outliers)
