"""
Input is a video

Counts detected markers. Rejected markers are cropped out and enhanced before marker detection is attempted again. 

Output is metrics on marker detection

"""


import os
from pathlib import Path
import time
import joblib
from PIL import Image
import sys

# necessary for importing SR, DN and Model module when running the script independently
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# qiets TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import cv2
from cv2 import aruco as aruco
from sklearn import svm
import tensorflow as tf

import Model
import DN
import SR


# seems necessary to avoid crashing the model
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# settings
INPUT_VIDEO = Path("evaluation_images/cage_hover.mp4")
TAG_PADDING = 0.3 # percentage size of tag added as padding when searching forn new
ASPECT_RATIO_DEVIATION = 0.7 # percentage similarity of a 1:1 ratio. images outside of threshhold is rejected

# save wrongly rejected markers
# OUTPUT_FRAMES = Path("evaluation_images/rejected") # save frame if any tag is detected, use Path object
OUTPUT_FRAMES = None
IMAGE_FORMAT = "png"

# save a video of the detection recording
OUTPUT_VIDEO = None
# OUTPUT_VIDEO = Path("evaluation_images")

# save metrics to spreadsheet
OUTPUT_SPREADSHEET = None
# OUTPUT_SPREADSHEET = Path("evaluation/spreadsheets")

# classifier model settings
CLASSIFIER_MODEL = Path("evaluation/classifier_models/2021-05-15_17-35-14.joblib")
FFT_DIMS = (16, 16)

# enhancer model settings
ENHANCER_NAME = "SR" # DN, SR, SRDN, DNSR, 1to1, resize
DN_ENHANCER_MODEL = Path("saved_models/256_20210515-221533_best")
SR_ENHANCER_MODEL = Path("saved_models/SR/128_20210515-104132_Final")
ENHANCER_INPUT_SIZE = 128
ENHANCER_OUTPUT_SIZE = 256 # pixel height of enhancement algorithm


# aruco parameters 
ARUCO_DICT = aruco.DICT_6X6_250
aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
arucoParameters = aruco.DetectorParameters_create() # default values
font = cv2.FONT_HERSHEY_PLAIN

""" helper functions """
# creates a histogram if a image
def find_image_features(input_img, dims):
    # read image, find shape
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dims)
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_norm = dft[:,:,0] / dft[:,:,0].size
    return dft_norm.flatten()

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


def make_folder(path):
    try:  
        # creating a folder 
        if not os.path.exists(path): 
            os.makedirs(path) 
    # if not created then raise error 
    except OSError: 
        print (f'Error: Creating directory of {path}') 


""" main """

#make output folders
if OUTPUT_SPREADSHEET is not None:
    make_folder(Path(f"{OUTPUT_SPREADSHEET}/{INPUT_VIDEO.stem}"))
if OUTPUT_FRAMES is not None:
    saved_tags_output_folder = Path(f"{OUTPUT_FRAMES}/{INPUT_VIDEO.stem}")
    make_folder(saved_tags_output_folder)

# create video reader object
cap = cv2.VideoCapture(str(INPUT_VIDEO)) # read video file

# set up recording
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = round(cap.get(cv2.CAP_PROP_FPS))

# initiate video capture object
if OUTPUT_VIDEO is not None:
    output_video_size = (width, (height + ENHANCER_INPUT_SIZE))
    vid_output_name = Path(f"{OUTPUT_VIDEO}/{INPUT_VIDEO.stem}_{ENHANCER_NAME}.mp4")
    video_cap = cv2.VideoWriter(
        str(vid_output_name),
        cv2.VideoWriter_fourcc(*"mp4v"), 
        (fps // 2), 
        output_video_size,
        )

# metrics
tags_found = 0
extra_ids_total = 0
enhance_attempt_total = 0
frame_count = 0
n_skipped = 0
realtime_fps = 0
saved_aspect_ratios = []
frame_times = []
spreadsheet_export = []

# initiate classifer model
classifier = joblib.load(CLASSIFIER_MODEL)

# initiate enhancement model
enhance_scale = ENHANCER_OUTPUT_SIZE / ENHANCER_INPUT_SIZE
# if ENHANCER_NAME:
#     sr_enhancer = Model.load_model(SR_ENHANCER_MODEL)
# elif ENHANCER_NAME == "DN":
#     dn_enhancer = Model.load_model(DN_ENHANCER_MODEL)
# elif (ENHANCER_NAME == "SRDN") or (ENHANCER_NAME == "DNSR"):
sr_enhancer = Model.load_model(SR_ENHANCER_MODEL)
dn_enhancer = Model.load_model(DN_ENHANCER_MODEL)


# main loop
video_time_start = time.time()
while(True):
    # metrics
    frame_time_start = time.time()
    n_ids = 0 # number of detected ids in session
    extra_ids = 0 # markers detected through enhancement in current frame
    candidate_max_height = 0
    enhance_attempt_frame = 0

    """ first marker detection cycle """
    # capture next frame and convert to grayscale
    ret, frame = cap.read()
    if ret != True: # break if last frame
    # if frame is None: # break if last frame
        break
    
    # add padding on top of frame, the area is used to display enhanced tags
    frame = cv2.copyMakeBorder(
        frame, 
        ENHANCER_INPUT_SIZE,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
        )

    # feed grayscaled video frame into tag detection algorithm
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    
    # count the number of tags identified:
    if ids is not None:
        n_ids = len(ids)
        tags_found += n_ids

    # crop out rejected tags (if there are any)
    if rejectedImgPoints is not None:
        rejected_img_matrices = []
        rejected_img_features = []
        rejected_img_position = []

        # iterate the the list of rejected tags
        for i, tag in enumerate(rejectedImgPoints):
            
            # iterate through each tag
            for j, square in enumerate(tag):

                # find coordinates defining the tag
                x_max = max(square[:, 1])
                x_min = min(square[:, 1])
                y_max = max(square[:, 0])
                y_min = min(square[:, 0])

                # set crop margin
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

                # crop candidate from current frame
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
                    n_skipped += 1
                    continue
                else:
                    saved_aspect_ratios.append(aspect_ratio)

                # find image features and add to stack to be classified
                rejected_img_matrices.append(tag_im)
                rejected_img_features.append(find_image_features(tag_im, FFT_DIMS))
                rejected_img_position.append([tag])


        # if frame offers any rejected tags
        if len(rejected_img_features) > 0:
            img_classifications = classifier.predict(rejected_img_features)

            # enhance each candidate to classify as either false negative or true negative
            x_offset = 50
            y_offset = 30

            # find images classified as a valid tag and iterate list
            valid_indexes = [i for i, e in enumerate(img_classifications) if e == 1]
            for i, idx in enumerate(valid_indexes):
                img = rejected_img_matrices[idx]
                        
                # save candidate if desired
                if OUTPUT_FRAMES is not None:                
                    name = Path(f"{saved_tags_output_folder}/{round(frame_count // fps)}-{round(frame_count % fps)}_{i}.{IMAGE_FORMAT}")
                    cv2.imwrite(str(name), img)
                    n_saved += 1

                # add padding to make it fit the model input size
                padded_img, pad = add_padding(img, ENHANCER_INPUT_SIZE)

                # enhance cropped image
                if ENHANCER_NAME == "SR":
                    enhanced_img = SR.predict(sr_enhancer, padded_img)
                    # enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_GRAY2BGR)

                elif ENHANCER_NAME == "DN":
                    enhanced_img = DN.predict(dn_enhancer, padded_img)
                    # enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_GRAY2BGR)

                elif ENHANCER_NAME == "SRDN":
                    enhanced_img, _ = DN.predict_model(dn_enhancer, SR.predict(sr_enhancer, padded_img))
                    # enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_GRAY2BGR)

                elif ENHANCER_NAME == "DNSR":
                    enhanced_img = SR.predict(dn_enhancer, DN.predict(sr_enhancer, padded_img))
                    # enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_GRAY2BGR)

                elif ENHANCER_NAME == "1to1": # convert image from numpy -> PIL (greyscale)
                    enhanced_img = Image.fromarray(padded_img).convert("L")

                elif ENHANCER_NAME == "resize": # convert image from numpy -> PIL (resize, greyscale)
                    enhanced_img = Image.fromarray(padded_img).convert("L").resize([round(x * enhance_scale) for x in reversed(padded_img.shape[:2])])
                    
                enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_GRAY2BGR)

                # retry marker detection
                enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
                enhanced_corners, enhanced_ids, _ = aruco.detectMarkers(
                    enhanced_gray, aruco_dict, parameters=arucoParameters)
                enhance_attempt_frame += 1
                enhance_attempt_total += 1
                
                if enhanced_ids is not None:
                    if len(enhanced_ids) != 1:
                        print(f"Error: Detected {len(enhanced_ids)} markers in enhanced image")
                    else:
                        # draw the detected tags on the current frame and cropped out tag
                        extra_ids += 1
                        extra_ids_total += 1
                        enhanced_img = aruco.drawDetectedMarkers(enhanced_img, enhanced_corners, ids=enhanced_ids, borderColor=(255,111,255))
                        frame = aruco.drawDetectedMarkers(frame, rejected_img_position[idx], ids=enhanced_ids, borderColor=(255,111,255))

                # crop enhanced candidate tag
                enhanced_img = enhanced_img[
                    round(pad["top"]*enhance_scale):round(pad["bottom"]*enhance_scale), 
                    round(pad["left"]*enhance_scale):round(pad["right"]*enhance_scale+15)
                    ]

                # draw enhanced candidate tag onto frame
                # check pasted tags are not out bounds of frame
                if enhanced_img.shape[0] > candidate_max_height:
                    candidate_max_height = enhanced_img.shape[0]

                if (width - x_offset < (enhanced_img.shape[1] + 10)):
                    y_offset += candidate_max_height + 10
                    x_offset = 50
                    candidate_max_height = 0

                frame[y_offset:y_offset + enhanced_img.shape[0], x_offset:x_offset + enhanced_img.shape[1]] = enhanced_img
                x_offset += enhanced_img.shape[1] + 10

    # draw detected aruco tags on output frame
    frame = aruco.drawDetectedMarkers(frame, corners, ids=ids)
    frame = aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(0, 0, 255))

    # draw text on output frame
    cv2.putText(frame, "Enhanced tags:", (10, 30), font, 2, (255,111,255), 1, cv2.LINE_AA)

    realtime_fps = (1 / (time.time() - frame_time_start)) * 0.5 + realtime_fps * 0.5
    cv2.putText(frame, f"FPS: {realtime_fps:.0f}", (10, (frame.shape[0] - 90)), font, 4, (255,111,255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Tags: {n_ids:02d}+{extra_ids:02d}={n_ids+extra_ids:02d}", (10, (frame.shape[0] - 40)), font, 4, (255,111,255), 2, cv2.LINE_AA)

    """ present final frame """
    # save video recording
    if OUTPUT_VIDEO is not None:
        video_cap.write(frame)

    # display frame
    cv2.imshow('Display', frame)
    
    # quit if user press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # end of video loop cleanup
    frame_count += 1
    frame_time= time.time() - frame_time_start
    frame_times.append(frame_time)

    if OUTPUT_SPREADSHEET is not None:
        spreadsheet_export.append({
            "n": frame_count,
            "time": frame_time,
            "ids": n_ids,
            "e_ids": extra_ids,
            "rejected": len(rejectedImgPoints),
            "false_rejects": enhance_attempt_frame
        })

# time measurements
total_time = time.time() - video_time_start
frames_per_sec = sum(frame_times) / frame_count

# opencv cleanup
cap.release()
if OUTPUT_VIDEO is not None:
    video_cap.release()
cv2.destroyAllWindows()

# print metrics
print(f"Input video size {width}x{height} @ {fps}FPS")
print(f"Tags detected total: {tags_found}")
print(f"Tags detected per frame: {tags_found / frame_count:.3f}")
print(f"Extra tags detected with enhancement total: {extra_ids_total}")
print(f"Extra tags detected per enhancement: {extra_ids_total / enhance_attempt_total:.3f}. {enhance_attempt_total} attempts total")
print(f"Average {frames_per_sec:.3f} s per frame ({1 / frames_per_sec:.3f} Hz)")
print(f"Skipped {n_skipped} images because size was larger than {ENHANCER_INPUT_SIZE}x{ENHANCER_INPUT_SIZE} or aspect ratio outside of bounds")
# print(f"{n_saved} files written to \"{saved_tags_output_folder}\"")

if OUTPUT_SPREADSHEET is not None:
    dataframe = pd.DataFrame(spreadsheet_export)
    dataframe.to_csv(Path(f"{OUTPUT_SPREADSHEET}/{INPUT_VIDEO.stem}/{ENHANCER_NAME}.csv"))
    print(f"Saved \"{ENHANCER_NAME}.csv\" to \"{OUTPUT_SPREADSHEET}/{INPUT_VIDEO.stem}\"")

if OUTPUT_VIDEO is not None:
    print(f"Video file saved as \"{vid_output_name}\"")