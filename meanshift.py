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

def mean_shift(image, bandwidth=3):
    data = image.reshape((-1, len(image[0])))
    clusters = data.copy()

    for i in tqdm(range(len(clusters))):
        x = clusters[i]

        while True:
            x_prev = x

            dists = np.sqrt(((x-data)**2).sum(axis=1))
            weights = (1 / (bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((dists / bandwidth)**2))
            # weights = [1 if dist <= bandwidth else 0 for dist in dists]
            tiled_weights = np.tile(weights, [len(x), 1]).transpose()
            
            weights_sum = sum(weights)
            x = np.multiply(tiled_weights, data).sum(axis=0) / max(weights_sum, 1)

            if euclidean_distance(x_prev, x) < .001:
                break

        clusters[i] = x

    centers = []
    labels = []

    num_clusters = 0
    for point in clusters:
        clustered = False
        for idx, center in enumerate(centers):
            if euclidean_distance(point, center) <= 2:
                labels.append(idx)
                clustered = True
                break
        if not clustered:
            centers.append(point)
            labels.append(num_clusters)
            num_clusters += 1

    return np.array(centers), np.array(labels)