"""
Inputs:
    - a superres model and its magnification coefficient
    - folder path to evaluation images

Outputs:
    - metrics on how many many tags that were detected
    - metrics on image quality

"""

import time
import glob
from PIL import Image
import sys
sys.path.append("/home/hakon/code/ACIT4630_SemesterProject")

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm

import Model

# parameters
MODEL_NAME = "saved_models/100_20210503-084117"
MODEL_INPUT_SIZE = 100
EVALUATION_IMAGE = "evaluation_images/isolated_tags/charuco_CH1_35-15_200_png"
IMAGE_FORMAT = "png"

# aruco parameters 
ARUCO_DICT = aruco.DICT_6X6_250 
aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
arucoParameters = aruco.DetectorParameters_create() # default values

# find images in folder
evaluation_images = glob.glob(EVALUATION_IMAGE + f"/*.{IMAGE_FORMAT}")
if len(evaluation_images) == 0:
    print(f"No .{IMAGE_FORMAT} files in \"{INPUT_FRAMES_FOLDER}\"")

# load model
model = Model.load_model(MODEL_NAME)

# count tags. assumes input is a greyscale image
def find_tags(input_image):
    # feed grayscale image into aruco-algorithm
    _, ids, _ = aruco.detectMarkers(
        input_image, aruco_dict, parameters=arucoParameters)
    
    # count the number of tags identified:
    if ids is not None:
        return len(ids)
    else:
        return 0

# metrics
n_ground_truth_detected = 0
n_hr_detected = 0
n_bicubic_detected = 0
n_images = 0

# predict and evaluate images
start_t = time.time()
for image_path in tqdm(evaluation_images[:100]):
# for image_path in evaluation_images:
    HR, LR, bicubic = Model.predict_model(model, image_path, MODEL_INPUT_SIZE)

    # convert images to openCV format and detect markers
    image = cv2.imread(image_path, 0)
    n_ground_truth_detected += find_tags(image)
    n_hr_detected += find_tags(np.array(HR))
    n_bicubic_detected += find_tags(np.array(bicubic))

    n_images += 1

# print evaluation
# n_images = len(evaluation_images)
print(f"Evaluated {n_images} images")
print("Detection rates:")
print("------------------------")
print(f"GT (.{IMAGE_FORMAT})\t{n_ground_truth_detected / n_images * 100:.2f} %")
print(f"HR\t\t{n_hr_detected / n_images * 100:.2f} %")
print(f"Bicubic\t\t{n_bicubic_detected / n_images * 100:.2f} %")
print("------------------------")
print(f"Finished in {time.time() - start_t:.1f} s")


