import os
import time
from pathlib import Path

import cv2 as cv

"""
Takes a video and creates frames as .jpg files according to parameters

"""

start_t = time.time()

# settings 
INPUT_VIDEO = Path("/home/wehak/Videos/charuco_CH3_35-15.mp4")
OUTPUT_PATH = Path(f"evaluation_images/frames/{INPUT_VIDEO.stem}")
SAVED_FRAMES_PER_SECOND = 2

# if output frames should be resized
RESIZE_OUTPUT = False
OUTPUT_HEIGHT = 256 # pixels (aspect ratio maintained unless crop is True)

# if output frame should be cropped to a square
SQUARE_CROP = False


# input_file_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]

cap = cv.VideoCapture(str(INPUT_VIDEO))
fps = round(cap.get(cv.CAP_PROP_FPS))

# creating a folder named data 
try: 
    if not os.path.exists(OUTPUT_PATH): 
        os.makedirs(OUTPUT_PATH) 
# if not created then raise error 
except OSError: 
    print (f'Error: Creating directory of {OUTPUT_PATH}') 


skip = SAVED_FRAMES_PER_SECOND * fps
currentframe = 0
# crop_offset = int((OUTPUT_WIDTH - OUTPUT_HEIGHT) / 2)

while(True):

    ret, frame = cap.read() # read next frame
    
    if ret: # if reading was succcessful
        if currentframe % skip == 0: # set with SAVED_FRAMES_PER_SECOND

            frame_height, frame_width, frame_layers = frame.shape
            aspect_ratio = frame_width / frame_height

            if RESIZE_OUTPUT:
                frame = cv.resize(frame, dsize=(round(OUTPUT_HEIGHT * aspect_ratio), OUTPUT_HEIGHT))
            
            if SQUARE_CROP:
                frame_height, frame_width, frame_layers = frame.shape
                crop_offset = int((frame_width - frame_height) / 2)
                frame = frame[:, crop_offset:crop_offset+frame_height]

            # write file 
            name = f"{OUTPUT_PATH}/{INPUT_VIDEO.stem}-{round(currentframe/fps)}.jpg"
            print("Creating... " + name)
            cv.imwrite(name, frame)

        currentframe += 1
    else:
        break

print(f"Completed in {time.time() - start_t :.2f} s")

#cleanup
cap.release()
cv.destroyAllWindows()