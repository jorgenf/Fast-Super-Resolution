import os
# import glob
# from PIL import Image
from pathlib import Path
import time
import joblib

import numpy as np
import cv2
from cv2 import aruco as aruco
from sklearn.linear_model import LogisticRegression
from sklearn import svm


"""
Input is a video

Detects and crops tags/markers every N'th second. Adds padding if necessary. Saves cropped marker to output path

Output is images of spesified size

"""

# settings
INPUT_VIDEO = Path("/home/wehak/Videos/ch1_fading.mp4")
# INPUT_VIDEO = Path("/home/wehak/Videos/charuco_CH1_35-15.mp4")
OUTPUT_FRAMES = Path("evaluation_images/classified_tags") # save frame if any tag is detected
TAG_PADDING = 0.1 # percentage size of tag added as padding when searching forn new
ASPECT_RATIO_DEVIATION = 0.6 # percentage similarity of a 1:1 ratio. images outside of threshhold is rejected
IMAGE_FORMAT = "png"
CLASSIFIER_MODEL = Path("evaluation/logistic_models/2021-05-14_20-10-42.joblib")
FFT_DIMS = (8, 8)
# CANNY_FILTER_SIZE = 3



# aruco parameters 
ARUCO_DICT = aruco.DICT_6X6_250
aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
arucoParameters = aruco.DetectorParameters_create() # default values
font = cv2.FONT_HERSHEY_PLAIN

#misc
input_file_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]

""" helper functions """
# creates a histogram if a image
def find_image_features(input_img, dims):
    features = []

    # # read image, find shape
    # img = cv2.imread(str(path), 0)
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dims)
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_norm = dft[:,:,0] / dft[:,:,0].size
    return dft_norm.flatten()



    # fft_sum = (np.sum(f.real) / f.size) / 255
    fshift = np.fft.fftshift(f).real
    # magnitude_spectrum = 20*np.log(np.abs(fshift))
    return fshift.flatten()


    # # return histogram
    # return magnitude_spectrum.flatten()
    # return list(x)

# # does Canny edge detection and counts the number of white pixels (edges)
# def count_edges(tag_im):
#     # count edges
#     edges = cv2.Canny(tag_im, 100, 200, apertureSize=CANNY_FILTER_SIZE)
#     edge_pixels = np.sum(edges == 255) / edges.size
#     return edge_pixels



# # create a output folder
# if OUTPUT_HEIGHT:
#     size_str = OUTPUT_HEIGHT
# else:
#     size_str = "x"

folder_name = Path(f"{OUTPUT_FRAMES}/test3_{IMAGE_FORMAT}")
try:  
    # creating a folder 
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name) 
# if not created then raise error 
except OSError: 
    print (f'Error: Creating directory of {folder_name}') 

# create video reader object
cap = cv2.VideoCapture(str(INPUT_VIDEO)) # read video file
# cap = cv2.VideoCapture(0) # capture video from default computer camera

# set up recording
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = round(cap.get(cv2.CAP_PROP_FPS))
# skip = int(SAVE_FRAME_EVERY_N_SECONDS * fps)

# metrics
tags_found = 0
frame_count = 0
# n_saved = 0
saved_aspect_ratios = []

# initiate classifer
classifier = joblib.load(CLASSIFIER_MODEL)


# main loop
while(True):
    n_ids = 0 # number of detected ids in session

    # capture next frame and convert to grayscale
    ret, frame = cap.read()
    if frame is None: # break if last frame
        break

    # feed grayscaled video frame into tag detection algorithm
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    
    # count the number of tags identified:
    if ids is not None:
        n_ids = len(ids)
        tags_found += n_ids

    # crop out tags and make images every N'th frame
    if rejectedImgPoints is not None:
        rejected_img_matrices = []
        rejected_img_features = []
        # iterate the the list of valid tags detected by openCV aruco
        for i, tag in enumerate(rejectedImgPoints):
            
            # iterate through each tag
            for j, square in enumerate(tag):

                # find coordinates defining the tag
                x_max = max(square[:, 1])
                x_min = min(square[:, 1])
                y_max = max(square[:, 0])
                y_min = min(square[:, 0])

                # add desired padding
                x_pad = (x_max - x_min) * TAG_PADDING
                y_pad = (y_max - y_min) * TAG_PADDING

                # map padded tags to pixel coordinates
                x1 = int(x_min - x_pad)
                y1 = int(y_min - y_pad)
                x2 = int(x_max + x_pad)
                y2 = int(y_max + y_pad)

                #check that no coordinate is outside of the image
                x1 = x1 if x1 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                x2 = x2 if x2 < width else width
                y2 = y2 if y2 < height else height

                # assign coordinates to image map
                top_left = (y1, x1)
                bottom_right = (y2, x2)

                # crop image
                tag_im = frame[x1:x2, y1:y2]

                # discard if aspect ratio is crazy
                aspect_ratio = tag_im.shape[1] / tag_im.shape[0]
                if (aspect_ratio < (1 * ASPECT_RATIO_DEVIATION)) or (aspect_ratio > (1 * (1 / ASPECT_RATIO_DEVIATION))):
                    continue
                else:
                    saved_aspect_ratios.append(aspect_ratio)

                # calculate histogram and add to stack of images to be classified
                rejected_img_matrices.append(tag_im)
                rejected_img_features.append(find_image_features(tag_im, FFT_DIMS))


        # classify as either false negative or true negative
        # sklearn
        img_classifications = classifier.predict(rejected_img_features)
        # print(img_classifications)
        x_offset = 20
        y_offset = 70
        valid_indexes = [i for i, e in enumerate(img_classifications) if e == 1]
        for i, idx in enumerate(valid_indexes):
            img = rejected_img_matrices[idx]
            frame[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
            x_offset += 50


                    
            # # save image                
            # name = Path(f"{folder_name}/{round(frame_count // fps)}-{round(frame_count % fps)}_{i}.{IMAGE_FORMAT}")
            # cv2.imwrite(str(name), img)
            # n_saved += 1
            # print(f"Created {name} of size {tag_im.shape[0]}x{tag_im.shape[1]}")
            # """
            # draw rectangle and ID on frame    
            # DO NOT USE other than for debugging. will create drawings on output images
            # cv2.putText(frame, f"id: {ids[i][0]}", top_left, font, 1, (0,0,255), 1, cv2.LINE_AA)
            # frame = cv2.rectangle(frame, top_left, bottom_right, (255,111,255), 1)
    
    # draw detected aruco tags
    # frame = aruco.drawDetectedMarkers(frame, corners, ids=ids)
    frame = aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(0, 0, 255))
    cv2.putText(frame, f"tags: {n_ids}", (10, 50), font, 4, (255,111,255), 2, cv2.LINE_AA)

    # display frame
    cv2.imshow('Display', frame)
    
    # quit if user press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # while loop cleanup
    frame_count += 1

    time.sleep(0.050)

cap.release()
cv2.destroyAllWindows()

print(f"Input video size {width}x{height} @ {fps}FPS")
print(f"Tags detected total: {tags_found}")
print(f"Tags detected per frame: {tags_found / frame_count:.3f}")
# print(f"{n_saved} files written to \"{folder_name}\"")

"""
# prints report on output image size
if OUTPUT_HEIGHT is None:
    frame_paths = glob.glob(folder_name + f"/*.{IMAGE_FORMAT}")
    if len(frame_paths) == 0:
        print(f"No .{IMAGE_FORMAT} files in {folder_name}")

    sizes = [Image.open(f, 'r').size for f in frame_paths]
    print(f"Largest output image is {max(sizes)} and smallest is {min(sizes)}")
else:
    print(f"Output image size is {OUTPUT_HEIGHT}x{OUTPUT_HEIGHT}")

# reprort on AR
print(f"Aspect ratio vary from {min(saved_aspect_ratios):.2f} to {max(saved_aspect_ratios):.2f}")
"""