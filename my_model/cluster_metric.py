import math
import numpy as np

from .distance_function import euclidean_distance

def silhouette_score(customers, labels):
    sum_score = 0
    cluster_list = cluster_grouping(customers, labels)
    
    for idx, cluster_i in enumerate(cluster_list):
        if len(cluster_i) == 1: continue

        for cus_i in cluster_i:
            a_i, b_i = 0, math.inf

            # Calculate a(i)
            for cus_j in cluster_i:
                a_i += euclidean_distance(cus_i, cus_j)
            a_i /= len(cluster_i) - 1

            # Calculate b(i)
            for idx2, cluster_j in enumerate(cluster_list):
                if idx == idx2: continue
                b_i_tmp = 0
                for cus_j in cluster_j:
                    b_i_tmp += euclidean_distance(cus_i, cus_j)
                b_i_tmp /= len(cluster_j)

                if b_i_tmp < b_i: b_i = b_i_tmp
            
            # Calculate s(i)
            s_i = (b_i - a_i) / max(a_i, b_i)
            sum_score += s_i
    
    return sum_score / len(customers)

def dunn_index(customers, labels):
    min_inter_dist = math.inf
    max_intra_dist = 0
    cluster_list = cluster_grouping(customers, labels)

    # Calculate minimum inter distance (using centroid distance)
    centroid_list = []
    for cluster in cluster_list:
        centroid_list.append(centroid(cluster))
    
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

def davies_bouldin_index(customers, labels):
    db_index = 0
    cluster_list = cluster_grouping(customers, labels)
    
    for i in range(len(cluster_list)):
        centroid_i = centroid(cluster_list[i])
        max_score = 0
        for j in range(len(cluster_list)):
            if i == j: continue
            centroid_j = centroid(cluster_list[j])
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

def centroid(cluster):
    return np.mean(cluster, axis=0)
