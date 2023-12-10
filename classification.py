import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path):
    return cv2.imread(image_path)


def ms_classify(input_img):
    img = input_img.reshape((-1, 3))

    bandwidth = 24
    #bandwidth = estimate_bandwidth(img, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(img)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    #plt.figure(1)
    #plt.clf()

    colors = ["#dede00", "#377eb8", "#f781bf"]
    markers = ["x", "o", "^"]

    '''
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(img[my_members, 0], img[my_members, 1], markers[k], color=col)
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    #plt.title("Estimated number of clusters: %d" % n_clusters_)
    #plt.show()
    '''

    labeled_img = labels.reshape(input_img.shape[0:2])
    for labelnum in range(len(labels_unique)):

        labelsN = (labeled_img==labelnum).astype(np.uint8)
        cv2.imshow('Label ' + str(labelnum), labelsN*255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = '/Users/jprice/cs283/proj/MeanShiftClassification/GORP.jpeg'
    gorp_img = load_image(image_path)
    ms_classify(gorp_img)