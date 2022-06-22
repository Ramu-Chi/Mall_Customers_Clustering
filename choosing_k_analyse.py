import pandas as pd

from my_model.k_mean import KMeans
from my_model.my_plot import *
from my_model.preprocessing import min_max_scaling, z_score_scaling

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# Scaling
X = z_score_scaling(customers)
X[:, 0] /= 3 # weight age = 1/3

print(FIELD)
print(X)

plot_elbow_method(X, 12)
plot_silhouette_method(X, 12)

# plot silhouette for k = 5
km = KMeans(5)
km.fit(X)
labels = km.predict(X)
plot_silhouette_single_k(X, labels)
