import numpy as np
import cv2
from tqdm import tqdm
import math


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def mean_shift(image, bandwidth=40, texture=False, kernel_flat=True, bandwidth_texture=40):
    data = image.reshape((-1, len(image[0]))).astype(float)
    clusters = data.copy()

    for i in tqdm(range(0, len(clusters))):
    
        while True:
            x = clusters[i]
            x_prev = x

            dists = np.sqrt(((x-data)**2).sum(axis=1))
            if kernel_flat:
                weights = [1 if dist <= bandwidth else 0 for dist in dists]
            else:
                weights = (1 / (bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((dists / bandwidth)**2))

            if texture:
                if kernel_flat:
                    weights_texture = [1 if dist <= bandwidth else 0 for dist in dists]
                else:
                    weights_texture = (1 / (bandwidth_texture*math.sqrt(2*math.pi))) * np.exp(-0.5*((dists / bandwidth_texture)**2))
                tiled_weights = np.concatenate((np.tile(weights, [len(x[:-1]), 1]).transpose(),
                                                np.tile(weights_texture, [1, 1]).transpose()), axis=-1)
            else:
                tiled_weights = np.tile(weights, [len(x), 1]).transpose()
            
            weights_sum = np.sum(weights)
            x = np.multiply(tiled_weights, data).sum(axis=0) / max(weights_sum, 1)

            if euclidean_distance(x_prev, x) < .01:
                break

            clusters[i] = x

    centers = []
    labels = []

    num_clusters = 0
    for point in clusters:
        clustered = False
        for idx, center in enumerate(centers):
            if euclidean_distance(point, center) <= 3:
                labels.append(idx)
                clustered = True
                break
        if not clustered:
            centers.append(point)
            labels.append(num_clusters)
            num_clusters += 1

    return np.array(centers), np.array(labels)