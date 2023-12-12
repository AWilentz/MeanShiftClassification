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
    data = image.reshape((-1, 3))
    clusters = data.copy()

    for i in tqdm(range(len(clusters))):
        x = clusters[i]

        while True:
            x_prev = x

            dists = np.sqrt(((x-data)**2).sum(axis=1))
            weights = (1 / (bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((dists / bandwidth)**2))
            tiled_weights = np.tile(weights, [len(x), 1])
            
            weights_sum = sum(weights)
            x_new = np.multiply(tiled_weights.transpose(), data).sum(axis=0) / weights_sum

            if euclidean_distance(x_prev, x_new) < .1:
                break

            x = x_new

        clusters[i] = x

    centers = []
    labels = []

    for point in clusters:
        num_clusters = 0
        clustered = False
        for idx, center in enumerate(centers):
            if euclidean_distance(point, center) <= 10:
                labels.append(idx)
                clustered = True
                break
        if not clustered:
            centers.append(point)
            labels.append(num_clusters)
            num_clusters += 1

    return np.array(centers), np.array(labels)