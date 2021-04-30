import os

import cv2
from cv2 import aruco as aruco

"""
Takes a video and counts the number of ArUco tags detected 

"""

# settings
INPUT_VIDEO = r"C:\Users\hweyd\Downloads\charuco_CH1_35-15.mp4"
OUTPUT_FRAMES = r"images\single_tags" # save frame if any tag is detected


# aruco parameters 
ARUCO_DICT = aruco.DICT_6X6_250
aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
arucoParameters = aruco.DetectorParameters_create() # default values
font = cv2.FONT_HERSHEY_PLAIN

#misc
input_file_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]

# create video reader object
cap = cv2.VideoCapture(INPUT_VIDEO) # read video file
# cap = cv2.VideoCapture(0) # capture video from default computer camera

# set up recording
# if OUTPUT_VIDEO is not None:
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, size)

# metrics
tags_found = 0
frame_count = 0

while(True):
    frame_count += 1
    n_ids = 0

    # capture next frame and convert to grayscale
    ret, frame = cap.read()
    if frame is None: # break if last frame
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # feed grayscale image into aruco-algorithm
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    
    # count the number of tags identified:
    if ids is not None:
        n_ids = len(ids)
        tags_found += n_ids

    # draw detectec aruco tags
    frame = aruco.drawDetectedMarkers(frame, corners, ids=ids)
    frame = aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(0, 0, 255))
    cv2.putText(frame, f"tags: {n_ids}", (10, 50), font, 4, (255,111,255), 2, cv2.LINE_AA)

    # display frame
    cv2.imshow('Display', frame)

    # saves frame if a tag is detected and parameter set
    if OUTPUT_FRAMES and ids is not None:
        # write file 
        name = f"{OUTPUT_FRAMES}/{input_file_name}-{round(frame_count)}.jpg"
        print("Creating... " + name)
        cv2.imwrite(name, frame)

    # save frame to video file
    if OUTPUT_VIDEO is not None:
        out.write(frame)
    
    # quit if user press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    break # debug

cap.release()
if OUTPUT_VIDEO is not None:
    out.release()
cv2.destroyAllWindows()

print(f"Tags detected total: {tags_found}")
print(f"Tags detected per frame: {tags_found / frame_count:.3f}")