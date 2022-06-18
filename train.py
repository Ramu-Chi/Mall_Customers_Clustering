from copy import deepcopy
import pandas as pd
import random

from my_model.k_mean import KMeans
from my_model.my_plot import plot_customer_clusters

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# # Random Sampling (avoid outliers)
# idx = random.sample(range(len(customers)), 100)
# customers = customers[idx, :]

X = deepcopy(customers)

# Normalization (min-max scaling)
min_i = [18, 15, 1]
max_i = [70, 137, 99]
for cus in X:
    for i in range(len(cus)):
        cus[i] = (cus[i] - min_i[i]) / (max_i[i] - min_i[i])
        if i == 0: cus[i] /= 3 # weight age = 1/3
        
print(FIELD)
print(X)

km = KMeans(k=5)
km.fit(X)
labels = km.predict(X)

plot_customer_clusters(customers, labels, FIELD)
