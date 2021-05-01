import os

import cv2
from cv2 import aruco as aruco


"""
Input is a video

Detects and crops tags/markers every N'th second. Adds padding if necessary. Saves cropped marker to output path

Output is images of spesified size

"""

# settings
INPUT_VIDEO = r"/home/hakon/code/GAN_pipeline/vids/charuco_36-18.mp4"
OUTPUT_FRAMES = r"evaluation_images/isolated_tags" # save frame if any tag is detected
SAVE_FRAME_EVERY_N_SECONDS = 3 # decimals for multiple saves per second
OUTPUT_HEIGHT = 500 # pixels (aspect ratio maintained unless crop is True)
TAG_PADDING = .5 # percentage size of tag added as padding



# aruco parameters 
ARUCO_DICT = aruco.DICT_6X6_250
aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
arucoParameters = aruco.DetectorParameters_create() # default values
font = cv2.FONT_HERSHEY_PLAIN

#misc
input_file_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]

# create a output folder
folder_name = f"{OUTPUT_FRAMES}/{input_file_name}_{OUTPUT_HEIGHT}"
try:  
    # creating a folder 
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name) 
# if not created then raise error 
except OSError: 
    print (f'Error: Creating directory of {folder_name}') 

# create video reader object
cap = cv2.VideoCapture(INPUT_VIDEO) # read video file
# cap = cv2.VideoCapture(0) # capture video from default computer camera

# set up recording
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = round(cap.get(cv2.CAP_PROP_FPS))
skip = int(SAVE_FRAME_EVERY_N_SECONDS * fps)

# metrics
tags_found = 0
frame_count = 0

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
    if (frame_count % skip == 0) and (ids is not None): # set with SAVE_FRAME_EVERY_N_SECONDS
        # iterate the the list of valid tags detected by openCV aruco
        for i, tag in enumerate(corners):
            
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

                # crops image if the tag is smaller than specified output size
                if tag_im.shape[0] > OUTPUT_HEIGHT:
                    center = tuple(x / 2 for x in tag_im.shape)
                    x = center[1] - OUTPUT_HEIGHT / 2
                    y = center[0] - OUTPUT_HEIGHT / 2
                    tag_im = tag_im[int(y):int(y + OUTPUT_HEIGHT), int(x):int(x + OUTPUT_HEIGHT )]

                # add padding if cropped image is smaller than desired output image size
                if tag_im.shape[0] < OUTPUT_HEIGHT:
        
                    # calculate and add necessary padding
                    top_pad = round((OUTPUT_HEIGHT - tag_im.shape[0]) / 2)
                    bottom_pad = OUTPUT_HEIGHT - (top_pad + tag_im.shape[0])
                    left_pad = round((OUTPUT_HEIGHT - tag_im.shape[1]) / 2)
                    right_pad = OUTPUT_HEIGHT - (left_pad + tag_im.shape[1])
                    
                    tag_im = cv2.copyMakeBorder(
                        tag_im, 
                        top_pad,
                        bottom_pad,
                        left_pad,
                        right_pad,
                        cv2.BORDER_CONSTANT)
                    
                # save image
                name = f"{OUTPUT_FRAMES}/{input_file_name}_{OUTPUT_HEIGHT}/{round(frame_count // fps)}-{round(frame_count % fps)}_{ids[i][0]}.jpg"
                cv2.imwrite(name, tag_im)
                # print(f"Created {name} of size {tag_im.shape[0]}x{tag_im.shape[1]}")

                # draw rectangle and ID on frame
                # DO NOT USE other than for debugging. will create drawings on output images
                # cv2.putText(frame, f"id: {ids[i][0]}", top_left, font, 1, (0,0,255), 1, cv2.LINE_AA)
                # frame = cv2.rectangle(frame, top_left, bottom_right, (255,111,255), 1)
    
    # draw detected aruco tags
    frame = aruco.drawDetectedMarkers(frame, corners, ids=ids)
    # frame = aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(0, 0, 255))
    cv2.putText(frame, f"tags: {n_ids}", (10, 50), font, 4, (255,111,255), 2, cv2.LINE_AA)

    # display frame
    cv2.imshow('Display', frame)
    
    # quit if user press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # while loop cleanup
    frame_count += 1


cap.release()
cv2.destroyAllWindows()

print(f"Input size {width}x{height} @ {fps}FPS")
print(f"Tags detected total: {tags_found}")
print(f"Tags detected per frame: {tags_found / frame_count:.3f}")