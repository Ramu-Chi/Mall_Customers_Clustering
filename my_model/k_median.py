import math
import numpy as np
import random

from .distance_function import euclidean_distance

def centroid(cluster):
    return np.median(cluster, axis=0)

class KMedians:
    def __init__(self, k, init_count=10, max_iter=300, tol=0.0001):
        self.k = k
        self.init_count = init_count
        self.max_iter = max_iter
        self.tol = tol

        self.centroid_list = None

    def fit(self, X):
        best_hist = None
        best_centroid_list = None
        min_error = math.inf

        # running k-means many times with different centroid init
        for i in range(self.init_count):
            hist = self.fit_single(X)
            if hist['SSE'][-1] < min_error:
                min_error = hist['SSE'][-1]
                best_hist = hist
                best_centroid_list = self.centroid_list
        
        self.centroid_list = best_centroid_list
        return best_hist
    
    def fit_single(self, X):
        history = {}
        history['SSE'] = []

        self.init_centroid(X)

        for i in range(self.max_iter):
            # Check Convergence Criterion
            error = self.sum_square_error(X)
            history['SSE'].append(error)

            if i > 0 and abs(history['SSE'][i] - history['SSE'][i - 1]) <= self.tol:
                return history

            # Assign customer to cluster
            cluster_list = self.get_cluster_list(X)

            # Re-compute centroid for each cluster
            for label, cluster in enumerate(cluster_list):
                if len(cluster) > 0: # bugfix: if cluster is empty then keep the previous centroid
                    self.centroid_list[label] = centroid(cluster)
        
        return history
    
    def predict(self, X):
        label_list = []
        for customer in X:
            label = self.get_label(customer)
            label_list.append(label)

        return np.array(label_list)
    
    def init_centroid(self, X):
        idx = random.sample(range(len(X)), self.k)
        self.centroid_list = X[idx, :]

    def sum_square_error(self, X):
        sum_square_error = 0
        cluster_list = self.get_cluster_list(X)
        for label, cluster in enumerate(cluster_list):
            centroid = self.centroid_list[label]
            for customer in cluster:
                sum_square_error += euclidean_distance(customer, centroid) ** 2
        
        return sum_square_error
    
    def get_cluster_list(self, X):
        cluster_list = []
        for i in range(self.k):
            cluster_list.append([])
        
        for customer in X:
            label = self.get_label(customer)
            cluster_list[label].append(customer)
        
        return cluster_list
    
    def get_label(self, customer):
        label = min(range(self.k), key=lambda i: euclidean_distance(customer, self.centroid_list[i]))
        return label
