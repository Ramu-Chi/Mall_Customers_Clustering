import math
import numpy as np

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
    
    def fit(self, X):
        self.centroid_list = self.init_centroid(self, X)

        for i in range(self.max_iter):
            # TODO: Check Convergence Criterion, neu thoa man thi break

            # TODO: Assign customer to cluster (closest to centroid in self.centroid_list)

            # TODO: Re-compute self.centroid_list
    
    # TODO:
    def init_centroid(self, X):
