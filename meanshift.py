# 1. Initialize random seed and window W.
# 2. Calculate the center of gravity (mean) of W.    
# 3. Shift the search window to the mean.    
# 4. Repeat Step 2 until convergence.

import numpy as np
import cv2
from tqdm import tqdm
import math
import concurrent.futures

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def mean_shift_iteration(data, clusters, i, bandwidth):
    x = clusters[i]

    while True:
        x_prev = x

        dists = np.sqrt(((x - data) ** 2).sum(axis=1))
        weights = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((dists / bandwidth) ** 2))
        # weights = [1 if dist <= bandwidth else 0 for dist in dists]
        tiled_weights = np.tile(weights, [len(x), 1]).transpose()

        weights_sum = sum(weights)
        x = np.multiply(tiled_weights, data).sum(axis=0) / max(weights_sum, 1)

        if euclidean_distance(x_prev, x) < .001:
            break

    return x


def mean_shift(image, bandwidth=3, threading=True):
    data = image.reshape((-1, len(image[0])))
    clusters = data.copy()

    if threading is False:
        counter = 0
        print("Starting cluster for loop without threading.")
        for i in range(len(clusters)):

            x = clusters[i]

            while True:
                x_prev = x

                dists = np.sqrt(((x-data)**2).sum(axis=1))
                weights = (1 / (bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((dists / bandwidth)**2))
                # weights = [1 if dist <= bandwidth else 0 for dist in dists]
                tiled_weights = np.tile(weights, [len(x), 1]).transpose()

                weights_sum = sum(weights)
                x = np.multiply(tiled_weights, data).sum(axis=0) / max(weights_sum, 1)

                if euclidean_distance(x_prev, x) < .01:
                    break

            clusters[i] = x

            counter += 1
            if counter % 500 == 0:
                print("Finished " + str(counter) + " out of " + str(len(clusters)))

    else:  # if threading is True

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            mean_shift_calls = {executor.submit(mean_shift_iteration, data, clusters, cluster, bandwidth): cluster
                                for cluster in range(len(clusters))}
            print("Starting cluster for loop with threading.")
            counter = 0
            for res in concurrent.futures.as_completed(mean_shift_calls):
                c = mean_shift_calls[res]
                try:
                    clusters[c] = res.result()
                except Exception as exc:
                    print('Generated an exception.')
                else:
                    counter += 1
                    if counter % 500 == 0:
                        print("Finished " + str(counter) + " out of " + str(len(clusters)))


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