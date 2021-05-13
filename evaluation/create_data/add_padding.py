"""
Adds padding to an image so it becomes a specified size. 
Necessary to to fit arbitrary sized images to a model with fixed inputs

"""



from pathlib import Path
import os

from tqdm import tqdm
import cv2

INPUT_FOLDER = Path("/home/wehak/code/ACIT4630_SemesterProject/evaluation_images/isolated_tags/ch2")
IMAGE_FORMAT = "jpg"
OUTPUT_FOLDER = Path("/home/wehak/code/ACIT4630_SemesterProject/evaluation_images/isolated_tags")
OUTPUT_HEIGHT = 250

# search for files in given folder
image_list = list(INPUT_FOLDER.glob(f"*.{IMAGE_FORMAT}"))
if len(image_list) == 0:
    print(f"No .{IMAGE_FORMAT}-files in '{INPUT_FOLDER}'")
    quit()
else:
    print(f"Found {len(image_list)} .{IMAGE_FORMAT}-files in '{INPUT_FOLDER}'")

# define folder name and create folder
folder_name = Path(f"{OUTPUT_FOLDER}/{INPUT_FOLDER.name}_{OUTPUT_HEIGHT}px-padding_{IMAGE_FORMAT}")
try: 
    # creating a folder 
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name) 
# if not created then raise error 
except OSError: 
    print (f'Error: Creating directory of {folder_name}') 

# iterate through images and add padding
for i, img_path in enumerate(image_list):
    img = cv2.imread(str(img_path))
    img_shape = img.shape

    # add vertical padding if cropped image is smaller than desired output image size
    if img.shape[0] < OUTPUT_HEIGHT:

        # calculate and add necessary padding
        top_pad = round((OUTPUT_HEIGHT - img.shape[0]) / 2)
        bottom_pad = OUTPUT_HEIGHT - (top_pad + img.shape[0])
        
        img = cv2.copyMakeBorder(
            img, 
            top_pad,
            bottom_pad,
            0,
            0,
            cv2.BORDER_CONSTANT)

    # add horizontal padding if cropped image is smaller than desired output image size
    if img.shape[1] < OUTPUT_HEIGHT:

        # calculate and add necessary padding
        left_pad = round((OUTPUT_HEIGHT - img.shape[1]) / 2)
        right_pad = OUTPUT_HEIGHT - (left_pad + img.shape[1])
        
        img = cv2.copyMakeBorder(
            img, 
            0,
            0,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT)
        
    # save image                
    name = Path(f"{folder_name}/{i}-{img_shape[0]}x{img_shape[1]}.{IMAGE_FORMAT}")
    cv2.imwrite(str(name), img)
    n_saved = i+1
    # print(f"Created {name} of size {img.shape[0]}x{img.shape[1]}")

print(f"Created {n_saved} images in folder '{folder_name}'")