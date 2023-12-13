import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from meanshift import mean_shift

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, '/Users/alexwilentz/MeanShift_py')

# Global variables
BANDWIDTH = 18 # 45 works decently with sklearn's mean shift
SUBSET_SIZE = 10000
CUSTOM = True


def load_image(image_path):
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    luv = cv2.cvtColor(rgb, cv2.COLOR_RGB2Luv)
    return luv


def census_transform(luv_img):
    # Drawn from https://stackoverflow.com/questions/37203970/opencv-grayscale-mode-vs-gray-color-conversion
    img_rgb = cv2.cvtColor(luv_img, cv2.COLOR_LUV2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    h, w = img_gray.shape
    census = np.zeros((h - 2, w - 2), dtype='uint8')

    cp = img_gray[1:h - 1, 1:w - 1]

    offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

    for u, v in offsets:
        census = (census << 1) | (img_gray[v:v + h - 2, u:u + w - 2] >= cp)

    census = np.pad(census, ((1, 1), (1, 1)), 'constant', constant_values=np.mean(census))

    return census


def highpass_filter(luv_img, sigma):
    img_rgb = cv2.cvtColor(luv_img, cv2.COLOR_LUV2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    h, w = img_gray.shape
    hp_filtered_img = img_gray - cv2.GaussianBlur(img_gray, (h - 1, w - 1), sigma)

    return hp_filtered_img

def lowpass_filter(luv_img, sigma):
    #img_rgb = cv2.cvtColor(luv_img, cv2.COLOR_LUV2RGB)
    img_gray = cv2.cvtColor(luv_img, cv2.COLOR_RGB2GRAY)

    h, w = img_gray.shape
    lp_filtered_img = cv2.GaussianBlur(img_gray, (h - 1, w - 1), sigma)

    return lp_filtered_img


def find_nearest_neighbor_label(point, labeled_points, labels):
    repeated_point = np.tile(point, (labeled_points.shape[0], 1))

    diff_between_points = repeated_point - labeled_points

    dists = np.linalg.norm(diff_between_points, axis=1)

    closest_point_idx = np.argmin(dists)
    closest_label = labels[closest_point_idx]
    return closest_label


def ms_classify(input_img, spatial=False):
    num_cols = input_img.shape[1]
    num_rows = input_img.shape[0]
    img = input_img.reshape((-1, 3))
    num_pixels = img.shape[0]

    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_LUV2RGB)
    img_rgb_cols = img_rgb.reshape((-1, 3))
    img = np.hstack((img, img_rgb_cols))

    # img[:,1] =img[:,1] * 255 / (np.max(img[:,1]) - np.min(img[:,1]))

    #m = np.mean(img[:,1])
    img[:,1] = 5*(img[:,1] - np.min(img[:,1]))

    sigma = np.sqrt((num_rows + num_cols) / 2) * 4
    hp_filtered_img = highpass_filter(input_img, np.sqrt((num_rows + num_cols) / 2) * 4)
    img = np.hstack((img, hp_filtered_img.reshape(-1, 1)))

    #hp_filtered_img = highpass_filter(input_img, 0.4)
    #img = np.hstack((img, hp_filtered_img.reshape(-1, 1)))

    #lp_filtered_img = lowpass_filter(input_img, np.sqrt((num_rows + num_cols) / 2) / 2)
    #img = np.hstack((img, lp_filtered_img.reshape(-1, 1)))

    vis_dims = True
    if vis_dims is True:
        cv2.imwrite('fig/u-gorp7.png', input_img[:, :, 1] * 255)
        cv2.imwrite('fig/v-gorp7.png', input_img[:, :, 2] * 255)
        cv2.imwrite('fig/r-gorp7.png', img_rgb[:, :, 0] * 255)
        cv2.imwrite('fig/g-gorp7.png', img_rgb[:, :, 1] * 255)
        cv2.imwrite('fig/b-gorp7.png', img_rgb[:, :, 2] * 255)
        cv2.imwrite('fig/hpf-gorp7.png', hp_filtered_img * 255)


    print("Added high pass filter dimension.")

    if spatial is True:  # approach 1
        col_val = np.array([np.arange(num_cols)])
        col_mat = np.repeat(col_val, num_rows, axis=0)
        col_col = col_mat.reshape((-1, 1))

        row_val = np.array([np.arange(num_rows)]).T
        row_mat = np.repeat(row_val, num_cols, axis=1)
        row_col = row_mat.reshape((-1, 1))

        img = np.hstack((img, row_col))
        img = np.hstack((img, col_col))

    # subsetting procedure
    subset_idxs = np.random.choice(img.shape[0], size=SUBSET_SIZE, replace=False)
    img_subset = img[subset_idxs, 1:]
    #img_subset=img
    #img = img_subset

    # bandwidth = estimate_bandwidth(img, quantile=0.2, n_samples=500)

    print("Starting subset clustering.")
    if CUSTOM is True:
        cluster_centers, labels = mean_shift(img_subset, bandwidth=BANDWIDTH)

    else:
        ms = MeanShift(bandwidth=BANDWIDTH, bin_seeding=True)
        ms.fit(img_subset)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # plt.figure(1)
    # plt.clf()

    colors = ["#dede00", "#377eb8", "#f781bf"]
    markers = ["x", "o", "^"]

    '''
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(img[my_members, 0], img[my_members, 1], markers[k], color=col)
        plt.plot(
            cluster_center[1],
            cluster_center[3],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    #plt.title("Estimated number of clusters: %d" % n_clusters_)
    #plt.show()
    '''

    labeled_img = np.zeros(num_pixels) - 1  # initialize all as -1
    labeled_img[subset_idxs] = labels

    for i in range(num_pixels):
        if labeled_img[i] == -1:
            point = img[i,1:]
            label = find_nearest_neighbor_label(point, img_subset, labels)
            labeled_img[i] = label


    labeled_img = labeled_img.reshape(input_img.shape[0:2])



    # cv2.imshow('Census', census)
    i = 1
    for labelnum in range(len(labels_unique)):

        labelsN = (labeled_img == labelnum).astype(np.uint8)

        kernel2 = np.ones((3, 3), np.uint8)
        erosion_mask = cv2.erode(labelsN, kernel2)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erosion_mask, connectivity=4)
        large_enough_objs = stats[stats[:, 4] > 100]
        num_objs = large_enough_objs[large_enough_objs[:, 4] < 2000, :].shape[0]
        cv2.imshow('Label: ' + str(labelnum) + ', Num objs: ' + str(num_objs), erosion_mask * 255)


        if num_objs > 0:
            print("Class " + str(i) + ": " + str(num_objs) + str(" objects"))
            i += 1

    # cv2.imshow('Census', census)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = 'data/gorp11.jpg'
    gorp_img = load_image(image_path)
    # highpass_filter(gorp_img)
    ms_classify(gorp_img)