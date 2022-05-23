import math
from webbrowser import get
import numpy as np
import random
import pandas as pd
from pyparsing import disable_diag

def distance(cus1, cus2):
    dist = (cus1[0] - cus2[0]) ** 2     # Gender
    dist += (cus1[1] - cus2[1]) ** 2    # Age
    dist += (cus1[2] - cus2[2]) ** 2    # Annual Income
    dist += (cus1[3] - cus2[3]) ** 2    # Spending Score
    return math.sqrt(dist)

def centroid(cluster):
    return np.mean(cluster, axis=0)

class KMeans:
    def __init__(self, k, n_init=10, max_iter=300, tol=0.0001):
        self.k = k
        # self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        self.centroid_list = None
        self.cluster_list = []
        self.inertia_ = 0
    
    def fit(self, X):
        self.centroid_list = self.init_centroid(X)

        for i in range(self.max_iter):
            # TODO: Assign customer to cluster (closest to centroid in self.centroid_list)
            self.cluster_list.clear()
            for j in range(self.k):
                cluster = []
                self.cluster_list.append(cluster)

            for x in X:
                closest_centroid = self.get_closest_centroid(x)
                self.cluster_list[closest_centroid].append(x)

            # TODO: Re-compute self.centroid_list
            for j in range(self.k):
                self.centroid_list[j] = centroid(self.cluster_list[j])

            # TODO: Check Convergence Criterion, neu thoa man thi break
            error = 0
            for j in range(self.k):
                for x in self.cluster_list[j]:
                    error += distance(x, self.centroid_list[j]) ** 2
            if abs(error - self.inertia_) <= self.tol:
                break
            self.inertia_ = error
    
    # TODO:
    def init_centroid(self, X):
        return random.sample(X, self.k)

    def get_closest_centroid(self, element):
        min_distance = distance(element, self.centroid_list[0])
        x_cluster = 0
        for i in range(1, self.k):
            dis = distance(element, self.centroid_list[i])
            if (dis < min_distance):
                min_distance = dis
                x_cluster = i
        return x_cluster

    def predict(self, X):
        x_cluster = []
        for x in X:
            x_cluster.append(self.get_closest_centroid(x))
        return x_cluster


if __name__ == "__main__":
    FIELD = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    gender_map = {0: 'Female', 1: 'Male'}

    # Prepare Data
    customer_df = pd.read_csv('Mall_Customers.csv')
    customer = customer_df[FIELD].to_numpy()

    for i in range(len(customer)):
        if customer[i][0] == 'Female':
            customer[i][0] = 0
        elif customer[i][0] == 'Male':
            customer[i][0] = 1
    
    kmeans = KMeans(5)
    list_cus = customer.tolist()
    kmeans.fit(list_cus)
    print(kmeans.predict(list_cus))
    print(kmeans.inertia_)