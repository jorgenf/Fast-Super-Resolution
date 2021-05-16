import time
import random
import glob
from PIL import Image
import sys
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("/home/wehak/code/ACIT4630_SemesterProject")

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import SR
import DN

"""
Inputs:
    - a superres model and its magnification coefficient
    - folder path to evaluation images

Outputs:
    - metrics on how many many tags that were detected
    - metrics on image quality
"""
def evaluate_model_single_tag(
    model_name, # model name and input size
    model_type,
    eval_im_folder, eval_im_format, # path to folder of evaluation images and their format (.jpg, .png etc)
    eval_sample_frac=1.0, # fraction of images in "eval_im_folder" checked
    verify_id = False
    ): 
    
    # aruco parameters 
    ARUCO_DICT = aruco.DICT_6X6_250 
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
    arucoParameters = aruco.DetectorParameters_create() # default values

    # find images in folder
    evaluation_images = [] 
    for im_format in eval_im_format:
        evaluation_images += glob.glob(f"{eval_im_folder}/*.{im_format}")
        if len(evaluation_images) == 0:
            print(f"No .{im_format} files in \"{eval_im_folder}\"")

    # load model
    model = SR.load_model(model_name)

    # count tags. assumes input is a greyscale image
    def find_tags(input_image, true_tag_id, input_im_name):
        # feed grayscale image into aruco-algorithm
        _, ids, _ = aruco.detectMarkers(
            input_image, aruco_dict, parameters=arucoParameters)
        
        # count the number of correct tags identified:
        if ids is not None:
            if (len(ids) == 1) and verify_id:
                if ids[0][0] == true_tag_id:
                    return 1
                else:
                    print(f"Warning: {input_im_name} wrongly identified ID {true_tag_id} as a {ids[0][0]}")
                    return 0
            elif len(ids) == 1:
                return 1
            elif len(ids) > 1:
                print(f"Warning: {input_im_name} identified more than 1 one ID: {len(ids)}")
                return 0
            else:
                print(f"Warning: {len(ids)}\n{ids}")
                return 0
        else:
            return 0

    # metrics
    n_ground_truth_detected = 0
    n_lr_detected = 0
    n_hr_detected = 0
    n_bicubic_detected = 0
    n_images = 0
    n_denoised_detected = 0
    n_original_detected = 0

    # predict and evaluate images
    random.shuffle(evaluation_images)
    start_t = time.time()
    for image_path in tqdm(evaluation_images[:int(eval_sample_frac * len(evaluation_images))]):
        # find tag id
        if verify_id:
            im_tag = int(image_path[image_path.rfind("_")+1 : image_path.rfind(".")])
        else:
            im_tag = 0

        im_name = Path(image_path).name

        # for image_path in evaluation_images:
        if model_type == "SR":
            HR, LR, bicubic = SR.predict_model(model, image_path)
            image = np.array(Image.open(image_path).convert("L"))
            n_ground_truth_detected += find_tags(image, im_tag, im_name)
            n_lr_detected += find_tags(np.array(LR), im_tag, im_name)
            n_hr_detected += find_tags(np.array(HR), im_tag, im_name)
            n_bicubic_detected += find_tags(np.array(bicubic), im_tag, im_name)
        elif model_type == "DN":
            denoised, original = DN.predict_model(model, image_path)
            n_denoised_detected += find_tags(np.array(denoised), im_tag, im_name)
            n_original_detected += find_tags(np.array(original), im_tag, im_name)
        else:
            raise Exception("No valid model type chosen.")
        # convert images to openCV format and detect markers
        # image = cv2.imread(image_path, 0)

        n_images += 1

    # print evaluation
    # n_images = len(evaluation_images)
    print(f"\nEvaluated {n_images} images")
    print("Detection rates:")
    print("------------------------")
    if model_type == "SR":
        print(f"Ground truth\t{n_ground_truth_detected / n_images * 100:.2f} %")
        print(f"LR\t\t{n_lr_detected / n_images * 100:.2f} %")
        print(f"HR\t\t{n_hr_detected / n_images * 100:.2f} %")
        print(f"Bicubic\t\t{n_bicubic_detected / n_images * 100:.2f} %")
    elif model_type == "DN":
        print(f"Original\t\t{n_original_detected / n_images * 100:.2f} %")
        print(f"Denoised\t\t{n_denoised_detected / n_images * 100:.2f} %")
    else:
        raise Exception("No valid model type chosen.")
    print("------------------------")
    print(f"Finished in {time.time() - start_t:.1f} s")


    with open(Path(f"{model_name}/assets/evaluation.txt"), 'w') as info:
        info.write(f"Evaluated {n_images} images\n")
        info.write("Detection rates:\n")
        info.write("------------------------\n")
        if model_type == "SR":
            info.write(f"Ground truth\t{n_ground_truth_detected / n_images * 100:.2f} %\n")
            info.write(f"LR\t\t{n_lr_detected / n_images * 100:.2f} %\n")
            info.write(f"HR\t\t{n_hr_detected / n_images * 100:.2f} %\n")
            info.write(f"Bicubic\t\t{n_bicubic_detected / n_images * 100:.2f} %\n")
        elif model_type == "DN":
            info.write(f"Original\t\t{n_original_detected / n_images * 100:.2f} %\n")
            info.write(f"Denoised\t\t{n_denoised_detected / n_images * 100:.2f} %\n")
        else:
            raise Exception("No valid model type chosen.")
        info.write("------------------------\n")
        info.write(f"Finished in {time.time() - start_t:.1f} s\n")
        info.close()

# run module independently
if __name__ == "__main__":
    # seems necessary to avoid crashing the model
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Enable/disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # example parameters
    model_name = Path("saved_models/SR/128_20210515-104132_Final")
    model_input_size = 128
    eval_im_folder = Path("evaluation_images/unrecognized_tags_256px-padding_png")
    eval_im_format = ("png", "jpg")

    # call evaluation function
    evaluate_model_single_tag(
        model_name, 
        "SR", 
        eval_im_folder, 
        eval_im_format,
        eval_sample_frac=0.1,
        )


# def evaluate_model_single_tag(
#     model_name, # model name and input size
#     model_type,
#     eval_im_folder, eval_im_format, # path to folder of evaluation images and their format (.jpg, .png etc)
#     eval_sample_frac=1.0, # fraction of images in "eval_im_folder" checked
#     verify_id = False
#     ): 