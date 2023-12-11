# 1. Initialize random seed and window W.
# 2. Calculate the center of gravity (mean) of W.    
# 3. Shift the search window to the mean.    
# 4. Repeat Step 2 until convergence.

import numpy as np
import cv2
from tqdm import tqdm
import math

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def mean_shift(image, bandwidth=28):
    data = image.reshape((-1, 3))
    cluster_centers = data.copy()
    for i in tqdm(range(len(cluster_centers))):
        x = cluster_centers[i]

        while True:
            dists = np.sqrt(((x-data)**2).sum(axis=1))
            weights = (1 / (bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((dists / bandwidth))**2)
            tiled_weights = np.tile(weights, [len(x), 1])

            weights_sum = sum(weights)
            new_x = np.multiply(tiled_weights.transpose(), data).sum(axis=0) / weights_sum

            if euclidean_distance(x, new_x) < .01:
                break

            x = new_x

        cluster_centers[i] = new_x

    return cluster_centers

image_path = "GORP_downsample.jpg"
image = cv2.imread(image_path)

cluster_centers = mean_shift(image)

with open("clusters.txt", 'w') as f:
    f.write(', '.join(str(item) for item in cluster_centers)+'\n')