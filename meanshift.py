# 1. Initialize random seed and window W.
# 2. Calculate the center of gravity (mean) of W.    
# 3. Shift the search window to the mean.    
# 4. Repeat Step 2 until convergence.

import numpy as np
import cv2
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def mean_shift(image, bandwidth=30):
    data = image.reshape((-1, 3))
    for i in range(len(data)):
        x = data[i]

        while True:
            mean_shift_vector = np.zeros_like(x).astype('float64')
            weights_sum = 0

            for j in range(len(data)):
                xi = data[j]
                distance = euclidean_distance(x, xi)

                weight = np.exp(-0.5 * (distance / bandwidth) ** 2)

                mean_shift_vector += weight * xi
                weights_sum += weight

            new_x = mean_shift_vector / weights_sum

            if euclidean_distance(x, new_x) < 1e-5:
                break

            x = new_x

        data[i] = new_x

    clustered_image = data.reshape(image.shape)

    return clustered_image

image_path = "GORP.jpeg"
image = cv2.imread(image_path)

# Perform mean shift clustering
clustered_image = mean_shift(image, bandwidth=30)

# Display the original and clustered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Mean Shift Clustering")
plt.imshow(clustered_image)