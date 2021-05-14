from pathlib import Path
import time
import random

import cv2
import numpy as np
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y

""" functions """

def create_dataset(valid_tags_path, rejected_tags_path, dims):

    # creates a histogram if a image
    def find_image_features(path):
        features = []

        # read image, find shape
        img = cv2.imread(str(path), 0)
        img = cv2.resize(img, dims)
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_norm = dft[:,:,0] / dft[:,:,0].size
        return dft_norm.flatten()
        # dft_shift = np.fft.fftshift(dft)
        # magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        # return magnitude_spectrum.flatten()
        # img = cv2.resize(img, dims)
        # f = np.fft.fft2(img)



        # # fft_sum = (np.sum(f.real) / f.size) / 255
        # fshift = np.fft.fftshift(f).real
        # # magnitude_spectrum = 20*np.log(np.abs(fshift))


        # # return histogram
        # return fshift.flatten()

    # finds images in folder and returns a list
    def find_images(path_str):
        input_path = Path(path_str)
        image_path_list = list(input_path.glob("*.png"))

        # quality control
        if len(image_path_list) == 0:
            print(f"Found 0 images in \"{input_path}\"")
            return None
        else:
            return image_path_list

    # input data folders
    valid_tags = find_images(valid_tags_path)
    rejected_tags = find_images(rejected_tags_path)

    t_start = time.time()

    x_data = []
    y_label = []
    for img_path in valid_tags:
        # hist, n_edges, fft_sum = find_image_features(img_path)
        x_data.append(find_image_features(img_path))
        y_label.append(1)

    for img_path in rejected_tags:
        # hist, n_edges, fft_sum = find_image_features(img_path)
        x_data.append(find_image_features(img_path))
        y_label.append(0)

    t_completed = time.time() - t_start

    # # normalize values to 0,1
    # x_data = np.array(x_data)
    # x_data = x_data / x_data.max(axis=0)


    # print(f"Processed {len(valid_tags) + len(rejected_tags)} images in {t_completed:.2f} s")
    return x_data, y_label


""" main """
if __name__ == "__main__":
    # create dataset
    valid = "evaluation_images/valid_tags/charuco_CH1_35-15_x_png"
    rejects = "evaluation_images/rejected_tags/charuco_CH1_35-15_sorted"

    X, y = create_dataset(valid, rejects, (4, 4))
    for i, x in enumerate(X[:5]):
        print(i, x, y[i])