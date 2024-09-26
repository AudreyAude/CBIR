import os
import cv2
import numpy as np
from descriptors import glcm , Bitdesc


def extract_features(image_path):
    """_summary_
    Extract features from a grayscale image
    Descriptors: GLCM, BiT, Haralick, etc.
    Args:
        image_path (_type_): Provide the relative path of the image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        features = glcm(img)
        return features
    else:
        return []


def process_datasets(root_folder):
    """_summary_
        Process each dataset folder with the root folder.
    Args:
        root_folder (_type_): root folder containing all the datasets
    """
    all_features = [] # List to store all features and metadata
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construct relative path and extract features
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                file_name = f'{relative_path.split("/")[0]}_{file}'
                features = extract_features(os.path.join(root, file))
                # Extract class name from the relative path
                class_name = os.path.basename(os.path.dirname(relative_path))
                # Append features, class name, and relative path to the list
                print(f'File: {file_name} -> Path: {relative_path} -> Features : {features}')
                all_features.append(features + [class_name, relative_path])
    signatures = np.array(all_features)
    np.save('signatures.npy', signatures)
    print('Features successfully stored.')
    return signatures

def main():
    signatures = process_datasets('../Dataset')
    print(signatures) 
    return signatures
     
if __name__ == '__main__':
    main()
