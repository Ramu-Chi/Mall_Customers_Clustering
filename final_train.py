import numpy as np
import pandas as pd

from my_model.cluster_metric import RMSSTD, davies_bouldin_index, dunn_index
from my_model.k_mean import KMeans
from my_model.my_plot import plot_customer_clusters, plot_train_history
from my_model.preprocessing import z_score_scaling

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# Remove Outliers
outlier_idx = [196, 197, 198, 199]
customers = np.delete(customers, outlier_idx, axis=0)

# Scaling
X = z_score_scaling(customers)
X[:, 0] /= 3 # weight age = 1/3

# print(FIELD)
# print(X)

# Training
km = KMeans(k=5)
hist = km.fit(X)
labels = km.predict(X)

# Showing Results
plot_train_history(hist['SSE'])

print('Davies-Bouldin Index: ', davies_bouldin_index(X, labels, model='kmean'))
print('RMSSTD: ', RMSSTD(X, labels, model='kmean'))
print('Dunn Index: ', dunn_index(X, labels, model='kmean'))

plot_customer_clusters(customers, labels, FIELD)
