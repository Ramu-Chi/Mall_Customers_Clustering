import numpy as np
from math import dist
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import random

from my_model.k_mean import KMeans

FIELD = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Prepare Data
customer_df = pd.read_csv('data/Mall_Customers.csv')
customers = customer_df[FIELD].to_numpy(dtype=object)

# Normalization (min-max scaling)
min_i = [18, 15, 1]
max_i = [70, 137, 99]
for cus in customers:
    for i in range(len(cus)):
        cus[i] = (cus[i] - min_i[i]) / (max_i[i] - min_i[i])
        if i == 0: cus[i] /= 3 # weight age = 1/3

# sort the centroid in lexical order from 1st attributes -> 3rd attributes
def sort_centroid(centroid_list):
    # sort_ind = np.lexsort((centroid_list[:, 2], centroid_list[:, 1], centroid_list[:, 0]))
    # return centroid_list[sort_ind]
    return np.array(sorted( centroid_list, key=(lambda x: dist(x, [0, 0, 0])) ))

k = 5
km = KMeans(k)
colors = cm.nipy_spectral([0, 1/5, 2/5, 3/5, 4/5])

# # Whole Dataset
# km.fit(customers)
# centroid_list_all_dataset = sort_centroid(km.centroid_list)

# plt.scatter(
#     centroid_list_all_dataset[:, 0], centroid_list_all_dataset[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
# )

# Random Sample 50% of Dataset 10 times
for i in range(10):
    idx = random.sample(range(len(customers)), 150)
    customers_sample = customers[idx, :]
    if i == 9: 
        customers_sample = np.append(customers_sample, [[0.3, 1, 1]], axis=0)
    km.fit(customers_sample)
    centroid_list = sort_centroid(km.centroid_list)
    # print([dist(centroid_list[i], centroid_list_all_dataset[i]) for i in range(k)])
    # print(centroid_list)
    plt.scatter(
        centroid_list[:, 0], centroid_list[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

plt.show()
