import numpy as np
import pandas as pd
import random

from my_model.cluster_metric import davies_bouldin_index, silhouette_score
from my_model.k_mean import KMeans
from my_model.my_plot import plot_customer_clusters, plot_train_history
from my_model.preprocessing import min_max_scaling, z_score_scaling

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# # Random Sampling (avoid outliers)
# idx = random.sample(range(len(customers)), 100)
# customers = customers[idx, :]

outlier_idx = [196, 197, 198, 199]
customers = np.delete(customers, outlier_idx, axis=0)

# Scaling
X = z_score_scaling(customers)
X[:, 0] /= 3 # weight age = 1/3

# print(FIELD)
# print(X)

km = KMeans(k=5)
hist = km.fit(X)
labels = km.predict(X)

plot_train_history(hist['SSE'])
print(davies_bouldin_index(X, labels))
plot_customer_clusters(customers, labels, FIELD)
