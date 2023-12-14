import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from meanshift import mean_shift

def load_image(image_path):
    rbg = cv2.imread(image_path)
    luv = cv2.cvtColor(rbg, cv2.COLOR_RGB2Luv)
    return luv


def highpass_filter(luv_img):
    img_rgb = cv2.cvtColor(luv_img, cv2.COLOR_LUV2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    h, w = img_gray.shape
    hp_filtered_img = img_gray - cv2.GaussianBlur(img_gray, (h-1, w-1), int(np.sqrt((h+w)/2)*4))

    return hp_filtered_img

def find_nearest_neighbor(point, labeled_points, labels):
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

    hp_filtered_img = highpass_filter(input_img)

    if spatial is True:
        col_val = np.array([np.arange(num_cols)])
        col_mat = np.repeat(col_val, num_rows, axis=0)
        col_col = col_mat.reshape((-1, 1))

        row_val = np.array([np.arange(num_rows)]).T
        row_mat = np.repeat(row_val, num_cols, axis=1)
        row_col = row_mat.reshape((-1, 1))

        img = np.hstack((img, row_col))
        img = np.hstack((img, col_col))

    img = np.hstack((img, hp_filtered_img.reshape(-1,1)))

    bandwidth = 16

    custom = True
    if custom is True:
        cluster_centers, labels = mean_shift(img, bandwidth=bandwidth, threading=True)

    else:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(img)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)


    labeled_img = labels.reshape(input_img.shape[0:2])
    i = 1
    for labelnum in range(len(labels_unique)):

        labelsN = (labeled_img==labelnum).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(labelsN, connectivity=4)
        large_enough_objs = stats[stats[:, 4] > 150]
        num_objs = large_enough_objs[large_enough_objs[:, 4] < 1000,:].shape[0]
        cv2.imshow('Label: ' + str(labelnum) + ', Num objs: ' + str(num_objs), labelsN*255)

        if num_objs > 0:
            print("Class " + str(i) + ": " + str(num_objs) + str(" objects"))
            i += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = '/Users/jprice/cs283/proj/MeanShiftClassification/GORP_downsample.jpeg'
    gorp_img = load_image(image_path)
    ms_classify(gorp_img)