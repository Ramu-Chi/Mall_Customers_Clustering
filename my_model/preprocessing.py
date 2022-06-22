from copy import deepcopy
import numpy as np

# Normalization
def min_max_scaling(X):
    min_i = np.min(X, axis=0)
    max_i = np.max(X, axis=0)

    normalized_X = deepcopy(X)
    for cus in normalized_X:
        for i in range(len(cus)):
            cus[i] = (cus[i] - min_i[i]) / (max_i[i] - min_i[i])

    return normalized_X

# Standardization
def z_score_scaling(X):
    standardized_X = deepcopy(X)
    standardized_X = standardized_X.astype(float)

    mean_i = np.mean(standardized_X, axis=0)
    std_i = np.std(standardized_X, axis=0)

    for cus in standardized_X:
        for i in range(len(cus)):
            cus[i] = (cus[i] - mean_i[i]) / std_i[i]
    
    return standardized_X
