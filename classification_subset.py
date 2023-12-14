import cv2
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import numpy as np
from meanshift import mean_shift
import time

# Global variables
BANDWIDTH = 40
SUBSET_SIZE = 8000
CUSTOM = True
IMAGE_PATH = 'data/gorp7.jpg'


def load_image(image_path):
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    luv = cv2.cvtColor(rgb, cv2.COLOR_RGB2Luv)
    return luv


def highpass_filter(luv_img, sigma):
    img_rgb = cv2.cvtColor(luv_img, cv2.COLOR_LUV2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    h, w = img_gray.shape
    hp_filtered_img = img_gray - cv2.GaussianBlur(img_gray, (h - 1, w - 1), sigma)

    return hp_filtered_img


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

    img[:,1] = 5*(img[:,1] - np.min(img[:,1]))


    hp_filtered_img = highpass_filter(input_img, np.sqrt((num_rows + num_cols) / 2) * 4)
    img = np.hstack((img, hp_filtered_img.reshape(-1, 1)))


    vis_dims = False
    if vis_dims is True:
        cv2.imshow('U*', input_img[:, :, 1] * 255)
        cv2.imshow('V*', input_img[:, :, 2] * 255)
        cv2.imshow('HPF image', hp_filtered_img * 255)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    print("Starting subset clustering.")
    t = time.time()

    if CUSTOM is True:
        cluster_centers, labels = mean_shift(img_subset, bandwidth=BANDWIDTH)

    else:
        ms = MeanShift(bandwidth=BANDWIDTH, bin_seeding=True)
        ms.fit(img_subset)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
    print("Clustering time: " + str(time.time() - t))

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)


    print("number of estimated clusters : %d" % n_clusters_)

    labeled_img_flat = np.zeros(num_pixels) - 1  # initialize all as -1
    labeled_img_flat[subset_idxs] = labels

    for i in range(num_pixels):
        if labeled_img_flat[i] == -1:
            point = img[i,1:]
            label = find_nearest_neighbor_label(point, img_subset, labels)
            labeled_img_flat[i] = label

    print("Clustering + nearest neighbors time: " + str(time.time() - t))

    labeled_img = labeled_img_flat.reshape(input_img.shape[0:2])

    color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (127, 0, 255), (0, 127, 255),
                  (255, 0, 127), (0, 255, 127), (255, 127, 0), (0, 127, 127), (255, 255, 0),
                  (255, 0, 255), (100, 200, 50)]

    plt.figure(figsize=(5,4))
    plt.clf()

    plot_features = False
    if plot_features is True:

        for k in range(len(labels_unique)):
            my_members = labeled_img_flat == k
            plt.plot(img[my_members, 2], img[my_members, -1], '.')
        plt.title("Clustering with 4000 points")
        plt.xlabel("v*")
        plt.ylabel("High pass filter value")
        plt.savefig("fig/fig3_4000_points.png")


    i = 1
    masks_list = []
    for labelnum in range(len(labels_unique)):

        labelsN = (labeled_img == labelnum).astype(np.uint8)

        kernel2 = np.ones((3, 3), np.uint8)
        erosion_mask = cv2.erode(labelsN, kernel2)
        masks_list.append(erosion_mask)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erosion_mask, connectivity=4)
        large_enough_objs = stats[stats[:, 4] > 80]
        num_objs = large_enough_objs[large_enough_objs[:, 4] < 2000, :].shape[0]
        cv2.imshow('Label: ' + str(labelnum) + ', Num objs: ' + str(num_objs), erosion_mask * 255)

        if num_objs > 0:
            print("Class " + str(i) + ": " + str(num_objs) + str(" objects"))
            i += 1

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_LUV2RGB)

    segmented_img = np.zeros((img_gray.shape[0], img_gray.shape[1], 3))

    for j in range(min([len(masks_list), len(color_list)])):
        mask = masks_list[j]
        segmented_img[mask==1] = color_list[j]

    cv2.imshow('Segmented Image', segmented_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gorp_img = load_image(IMAGE_PATH)
    ms_classify(gorp_img)