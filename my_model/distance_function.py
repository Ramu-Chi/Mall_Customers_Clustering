import math

def euclidean_distance(cus1, cus2):
    dist = 0
    attr_count = len(cus1)
    for i in range(attr_count):
        dist += (cus1[i] - cus2[i]) ** 2
    
    return math.sqrt(dist)

def manhattan_distance(cus1, cus2):
    dist = 0
    attr_count = len(cus1)
    for i in range(attr_count):
        dist += abs(cus1[i] - cus2[i])
    
    return dist
