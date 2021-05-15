import os
# import glob
# from PIL import Image
from pathlib import Path
import time
import joblib

import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("/home/wehak/code/ACIT4630_SemesterProject")

import numpy as np
import cv2
from cv2 import aruco as aruco
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import tensorflow as tf

import SR


# seems necessary to avoid crashing the model
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


"""
Input is a video

Detects and crops tags/markers every N'th second. Adds padding if necessary. Saves cropped marker to output path

Output is images of spesified size

"""

# settings
# INPUT_VIDEO = Path("/home/wehak/Videos/ch1_fading.mp4")
INPUT_VIDEO = Path("/home/wehak/Videos/charuco_CH1_35-15.mp4")
OUTPUT_FRAMES = Path("evaluation_images/valid_tags") # save frame if any tag is detected, use Path object
OUTPUT_VIDEO = None
# OUTPUT_VIDEO = Path("/home/wehak/Videos")
TAG_PADDING = 0.3 # percentage size of tag added as padding when searching forn new
ASPECT_RATIO_DEVIATION = 0.7 # percentage similarity of a 1:1 ratio. images outside of threshhold is rejected
IMAGE_FORMAT = "png"
# CLASSIFIER_MODEL = Path("evaluation/logistic_models/2021-05-14_20-10-42.joblib")
CLASSIFIER_MODEL = Path("evaluation/logistic_models/2021-05-15_14-14-24.joblib")
FFT_DIMS = (16, 16)
# CANNY_FILTER_SIZE = 3
    
ENHANCER_MODEL = Path("saved_models/SR/128_20210514-143939")
ENHANCER_INPUT_SIZE = 128
ENHANCER_OUTPUT_SIZE = 256 # pixel height of enhancement algorithm



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

def add_padding(img, output_size):
    pad = {
        "top" : 0,
        "bottom" : 0,
        "left" : 0,
        "right" : 0
    }
    # add vertical padding if cropped image is smaller than desired output image size
    if img.shape[0] < output_size:

        # calculate and add necessary padding
        pad["top"] = round((output_size - img.shape[0]) / 2)
        pad["bottom"] = output_size - (pad["top"] + img.shape[0])
        
        img = cv2.copyMakeBorder(
            img, 
            pad["top"],
            pad["bottom"],
            0,
            0,
            cv2.BORDER_REPLICATE
            # cv2.BORDER_CONSTANT,
            # value=(255, 255, 255)
            )

        pad["bottom"] = img.shape[0] - pad["bottom"]

    # add horizontal padding if cropped image is smaller than desired output image size
    if img.shape[1] < output_size:

        # calculate and add necessary padding
        pad["left"] = round((output_size - img.shape[1]) / 2)
        pad["right"] = output_size - (pad["left"] + img.shape[1])
        
        img = cv2.copyMakeBorder(
            img, 
            0,
            0,
            pad["left"],
            pad["right"],
            cv2.BORDER_REPLICATE
            # cv2.BORDER_CONSTANT,
            # value=(255, 255, 255)
            )
        
        pad["right"] = img.shape[1] - pad["right"]

    
    return img, pad


""" main """
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

folder_name = Path(f"{OUTPUT_FRAMES}/{ENHANCER_MODEL.name}_{IMAGE_FORMAT}")
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
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initiate video capture object
if OUTPUT_VIDEO is not None:
    vid_output_name = Path(f"{OUTPUT_VIDEO}/{INPUT_VIDEO.stem}_{ENHANCER_MODEL.name}.mp4")
    video_cap = cv2.VideoWriter(
        str(vid_output_name), 
        cv2.VideoWriter_fourcc(*"XVID"), 
        (fps // 2), 
        (width, (height + ENHANCER_OUTPUT_SIZE))
        )

# metrics
tags_found = 0
extra_tags_found = 0
enhance_attempt = 0
frame_count = 0
n_skipped = 0

# n_saved = 0
saved_aspect_ratios = []
video_time_start = time.time()
frame_times = []

# initiate classifer model
classifier = joblib.load(CLASSIFIER_MODEL)

# initiate enhancement model
enhancer = SR.load_model(ENHANCER_MODEL)


# main loop
while(True):
    frame_time_start = time.time()
    n_ids = 0 # number of detected ids in session
    extra_ids = 0 # markers detected through enhancement

    """ first marker detection cycle """
    # capture next frame and convert to grayscale
    ret, frame = cap.read()
    if frame is None: # break if last frame
        break
    
    # add padding on top of frame, the area is used to display enhanced tags
    frame = cv2.copyMakeBorder(
        frame, 
        ENHANCER_OUTPUT_SIZE,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
        )

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
                # top_left = (y1, x1)
                # bottom_right = (y2, x2)

                # crop image
                tag_im = frame[x1:x2, y1:y2]

                # QC: image size
                if max(tag_im.shape[:2]) > ENHANCER_INPUT_SIZE:
                    continue
                    n_skipped += 1
                if min(tag_im.shape[:2]) < min(FFT_DIMS):
                    continue
                    n_skipped += 1

                # QC: discard if aspect ratio is crazy
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
        if len(rejected_img_features) > 0:
            img_classifications = classifier.predict(rejected_img_features)
            # print(img_classifications)
            x_offset = 10
            y_offset = 0
            valid_indexes = [i for i, e in enumerate(img_classifications) if e == 1]
            for i, idx in enumerate(valid_indexes):
                img = rejected_img_matrices[idx]

                # print(max(img.shape))
                # frame[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

                # add padding to make it fit the model input size
                enhanced_img, pad = add_padding(img, (ENHANCER_OUTPUT_SIZE // 2))

                # enhance frame
                enhanced_img = SR.predict(enhancer, enhanced_img)
                enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_GRAY2BGR)
                # print(pad["top"]*2,pad["bottom"]*2, pad["left"]*2,pad["right"]*2)
                # enhanced_img = enhanced_img[pad["top"]*2:pad["bottom"]*2, pad["left"]*2:pad["right"]*2]

                if (width - x_offset < ENHANCER_OUTPUT_SIZE):
                    y_offset += ENHANCER_OUTPUT_SIZE
                    x_offset = 10

                # draw enhanced image on frame
                # frame[y_offset:y_offset + enhanced_img.shape[0], x_offset:x_offset + enhanced_img.shape[1]] = enhanced_img
                frame[y_offset:y_offset + ENHANCER_OUTPUT_SIZE, x_offset:x_offset + ENHANCER_OUTPUT_SIZE] = enhanced_img
                x_offset += ENHANCER_OUTPUT_SIZE
                # x_offset += max(img.shape) * 2 + 10

                enhance_attempt += 1
                        
                # # save image                
                name = Path(f"{folder_name}/{round(frame_count // fps)}-{round(frame_count % fps)}_{i}.{IMAGE_FORMAT}")
                cv2.imwrite(str(name), img)
                # n_saved += 1
                # print(f"Created {name} of size {tag_im.shape[0]}x{tag_im.shape[1]}")
                # """
                # draw rectangle and ID on frame    
                # DO NOT USE other than for debugging. will create drawings on output images
                # cv2.putText(frame, f"id: {ids[i][0]}", top_left, font, 1, (0,0,255), 1, cv2.LINE_AA)
                # frame = cv2.rectangle(frame, top_left, bottom_right, (255,111,255), 1)

    """ second marker detection cycle """
    # feed grayscaled video frame into tag detection algorithm
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners2, ids2, rejectedImgPoints2 = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)

    if ids2 is not None:
        extra_ids = len(ids2) - n_ids
        extra_tags_found += extra_ids

    # draw detected aruco tags
    frame = aruco.drawDetectedMarkers(frame, corners2, ids=ids2)
    frame = aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(0, 0, 255))
    cv2.putText(frame, f"tags: {n_ids} + {extra_ids}", (10, (frame.shape[0] - 40)), font, 4, (255,111,255), 2, cv2.LINE_AA)





    """ present final frame """
    # display frame
    cv2.imshow('Display', frame)
    
    # quit if user press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if OUTPUT_VIDEO is not None:
        video_cap.write(frame)

    # while loop cleanup
    frame_count += 1
    frame_times.append(time.time() - frame_time_start)

    # time.sleep(0.050)

cap.release()
if OUTPUT_VIDEO is not None:
    video_cap.release()
cv2.destroyAllWindows()

total_time = time.time() - video_time_start
frames_per_sec = sum(frame_times) / frame_count


print(f"Input video size {width}x{height} @ {fps}FPS")
print(f"Tags detected total: {tags_found}")
print(f"Tags detected per frame: {tags_found / frame_count:.3f}")
print(f"Extra tags detected with enhancement total: {extra_tags_found}")
print(f"Extra tags detected per enhancement: {extra_tags_found / enhance_attempt:.3f}. {enhance_attempt} attempts total")
print(f"Average {frames_per_sec:.3f} s per frame ({1 / frames_per_sec:.3f} Hz)")
print(f"Skipped {n_skipped} images because size was larger than {ENHANCER_INPUT_SIZE} px")
# print(f"{n_saved} files written to \"{folder_name}\"")

if OUTPUT_VIDEO is not None:
    print(f"Video file saved as \"{vid_output_name}\"")

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