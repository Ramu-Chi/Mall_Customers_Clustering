import numpy as np
import pandas as pd

from my_model.cluster_metric import RMSSTD, davies_bouldin_index, dunn_index
from my_model.k_mean import KMeans
from my_model.k_median import KMedians
from my_model.my_plot import plot_customer_clusters
from my_model.preprocessing import z_score_scaling

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# Scaling
X = z_score_scaling(customers)
X[:, 0] /= 3 # weight age = 1/3

# print(FIELD)
# print(X)

# Training K-means & K-medians
kmean = KMeans(k=5)
kmean.fit(X)

kmedian = KMedians(k=5)
kmedian.fit(X)

# Predicting with outliers removed 
outlier_idx = [196, 197, 198, 199]
X = np.delete(X, outlier_idx, axis=0)
outliers = customers[outlier_idx, :]
customers = np.delete(customers, outlier_idx, axis=0)

kmean_labels = kmean.predict(X)
kmedian_labels = kmedian.predict(X)

# Showing Results
print('Davies-Bouldin Index (K-means): ', davies_bouldin_index(X, kmean_labels, model='kmean'))
plot_customer_clusters(customers, kmean_labels, FIELD, title='K-means')

print('Davies-Bouldin Index (K-meadians): ', davies_bouldin_index(X, kmedian_labels, model='kmedian'))
plot_customer_clusters(customers, kmedian_labels, FIELD, title='K-medians')
