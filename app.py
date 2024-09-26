import numpy as np
from distances import retrieve_similar_images
from data_processing import extract_features
# Load offline signatures
signatures = np.load('signatures.npy')


def main():
    query_img = 'test_kth.png'
    features = extract_features(query_img)
    result = retrieve_similar_images(features_db=signatures, query_features=features, distance='euclidean', num_results=10)
    print(f'Results\n--------\n{result}')
    print(signatures)
    
if __name__ == '__main__':
    main()

