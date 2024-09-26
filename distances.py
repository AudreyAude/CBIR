import numpy as np
from scipy.spatial import distance

def manhattan_distance(v1, v2):
    """Compute the Manhattan distance between two vectors."""
    return np.sum(np.abs(np.array(v1, dtype=np.float64) - np.array(v2, dtype=np.float64)))

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((np.array(v1) - np.array(v2).astype('float'))**2))

def chebyshev_distance(v1, v2):
    """Compute the Chebyshev distance between two vectors."""
    return np.max(np.abs(np.array(v1, dtype=np.float64) - np.array(v2, dtype=np.float64)))
def canberra_distance(v1, v2):
    return distance.canberra(v1, v2)

def retrieve_similar_images(features_db, query_features, distance, num_results):
    
    distances = []
    
    for instance in features_db:
        features, label, img_path = instance[:-2],instance[-2], instance[-1]
        if distance == 'euclidean':
            dist = euclidean_distance(query_features, features)
        if distance == 'manhattan':
            dist = manhattan_distance(query_features, features)
        if distance == 'chebyshev':
            dist = chebyshev_distance(query_features, features)
        if distance == 'canberra':
            dist = canberra_distance(query_features, features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]
