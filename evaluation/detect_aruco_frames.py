
import glob
import time

import cv2
import cv2.aruco as aruco

"""
Opens a folder, finds all .jpg's and counts the number of ArUco tags detected

"""

start_t = time.time()

# settings
INPUT_FRAMES_FOLDER = "images/CH1_frames"
DISPLAY_FRAMES = True # False if there are more than 10 frames in folder
ARUCO_DICT = aruco.DICT_6X6_250 

# aruco parameters 
aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
arucoParameters = aruco.DetectorParameters_create() # default values
font = cv2.FONT_HERSHEY_PLAIN

frame_paths = glob.glob(INPUT_FRAMES_FOLDER + "/*.jp*g")
if len(frame_paths) == 0:
    print(f"No .jpg/.jpeg files in {INPUT_FRAMES_FOLDER}")

# metrics
tags_found = 0
frame_count = 0

for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Could not read {frame_path}")
        continue

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # feed grayscale image into aruco-algorithm
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    
    # count the number of tags identified:
    if ids is not None:
        tags_found += len(ids)

    if DISPLAY_FRAMES and len(frame_paths) < 10:
        # draw detected aruco tags and text
        frame = aruco.drawDetectedMarkers(frame, corners, ids=ids)
        frame = aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(0, 0, 255))
        cv2.putText(frame, f"tags: {len(ids)}", (10, 50), font, 4, (255,111,255), 2, cv2.LINE_AA)

        # display frame
        cv2.imshow('Display', frame)
        cv2.waitKey(0)

cv2.destroyAllWindows()

if len(frame_paths) > 0:
    print(f"Frames read: {frame_count}")
    print(f"Tags detected total: {tags_found} / {frame_count * 17}")
    print(f"Tags detected per frame: {tags_found / frame_count:.3f}")
    print(f"Completed in {time.time() - start_t :.2f} s")