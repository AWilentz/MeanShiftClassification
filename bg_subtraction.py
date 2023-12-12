import cv2
import matplotlib.pyplot as plt
import numpy as np

def bg_subtraction(img_path, bg_img_path):
    img = cv2.imread(img_path)
    bg_img = cv2.imread(bg_img_path)

    diff_img = np.maximum(img-bg_img, np.zeros_like(img))

    cv2.imshow('Difference img', diff_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_path = 'gorp1.jpeg'
    bg_img_path = 'gorp_bg.jpeg'
    bg_subtraction(img_path, bg_img_path)

