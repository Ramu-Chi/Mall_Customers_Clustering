import math
import numpy as np

from .distance_function import euclidean_distance
from .k_mean import centroid as k_mean_centroid
from .k_median import centroid as k_median_centroid

# calculate the average silhouette coefficient of all samples
def silhouette_score(customers, labels):
    coefficient_list = silhouette_coefficients(customers, labels)
    return np.mean(coefficient_list)

# calculate the silhouette coefficient of each sample
def silhouette_coefficients(customers, labels):
    coefficient_list = []
    cluster_list = cluster_grouping(customers, labels)

    for idx in range(len(customers)):
        a_i, b_i = 0, math.inf
        cus_i = customers[idx]
        label_i = labels[idx]

        if len(cluster_list[label_i]) == 1: 
            coefficient_list.append(0) # s_i = 0
            continue

        # Calculate a(i)
        for cus_j in cluster_list[label_i]:
            a_i += euclidean_distance(cus_i, cus_j)
        a_i /= len(cluster_list[label_i]) - 1

        # Calculate b(i)
        for label_j in range(len(cluster_list)):
            if label_i == label_j: continue
            b_i_tmp = 0
            for cus_j in cluster_list[label_j]:
                b_i_tmp += euclidean_distance(cus_i, cus_j)
            b_i_tmp /= len(cluster_list[label_j])

            if b_i_tmp < b_i: b_i = b_i_tmp
        
        # Calculate s(i)
        s_i = (b_i - a_i) / max(a_i, b_i)
        coefficient_list.append(s_i)
    
    return np.array(coefficient_list)

def RMSSTD(customers, labels, model='kmean'):
    centroid_func = get_centroid_func(model)

    cluster_list = cluster_grouping(customers, labels)
    sum_square_error = 0
    k = len(cluster_list)
    P = len(customers[0])

    for cluster_i in cluster_list:
        centroid_i = centroid_func(cluster_i)
        for cus in cluster_i:
            sum_square_error += euclidean_distance(cus, centroid_i) ** 2
    
    return math.sqrt( sum_square_error / (P * (len(customers) - k)) )

def dunn_index(customers, labels, model='kmean'):
    centroid_func = get_centroid_func(model)

    min_inter_dist = math.inf
    max_intra_dist = 0
    cluster_list = cluster_grouping(customers, labels)

    # Calculate minimum inter distance (using centroid distance)
    centroid_list = []
    for cluster in cluster_list:
        centroid_list.append(centroid_func(cluster))
    
    k = len(centroid_list)
    for i in range(k):
        for j in range(i + 1, k):
            inter_dist = euclidean_distance(centroid_list[i], centroid_list[j])
            min_inter_dist = min(inter_dist, min_inter_dist)
    
    # Calculate maximum intra distance (using 2 * mean error)
    for cluster in cluster_list:
        intra_dist = 2 * mean_error_single_cluster(cluster)
        max_intra_dist = max(intra_dist, max_intra_dist)
    
    # Calculate dunn index
    return min_inter_dist / max_intra_dist

def davies_bouldin_index(customers, labels, model='kmean'):
    centroid_func = get_centroid_func(model)

    db_index = 0
    cluster_list = cluster_grouping(customers, labels)
    
    for i in range(len(cluster_list)):
        centroid_i = centroid_func(cluster_list[i])
        max_score = 0
        for j in range(len(cluster_list)):
            if i == j: continue
            centroid_j = centroid_func(cluster_list[j])
            score = (mean_error_single_cluster(cluster_list[i]) + mean_error_single_cluster(cluster_list[j])) \
                    / euclidean_distance(centroid_i, centroid_j)
            max_score = max(score, max_score)
        
        db_index += max_score

    db_index /= len(cluster_list)
    return db_index

# Mean Error of single cluster (or mean distance to the centroid)
def mean_error_single_cluster(cluster):
    centroid = np.mean(cluster, axis=0)
    se = 0
    for cus in cluster:
        se += euclidean_distance(cus, centroid)

    return se / len(cluster)

def cluster_grouping(customers, labels):
    cluster_list = []
    for i in range(len(set(labels))):
        cluster_list.append([])

    for idx, label in enumerate(labels):
        cluster_list[label].append(customers[idx])
    
    return cluster_list

def get_centroid_func(model):
    if model == 'kmedian':
        return k_median_centroid
    else:
        return k_mean_centroid
