import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image
from pathlib import Path
import DN
import SR
import Data
import Model_util
from evaluation.util import evaluate_model_single_tag
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Enable/disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


evaluation_256_image_path = "evaluation_images/isolated_tags/charuco_CH1_35-15_256_png/14-0_10.png"
evaluation_128_image_path = "evaluation_images/isolated_tags/charuco_CH1_35-15_128_png/14-0_10.png"
eval_im_256_folder = Path("evaluation_images/isolated_tags/charuco_CH1_35-15_256_png")
eval_im_128_folder = Path("evaluation_images/isolated_tags/charuco_CH1_35-15_128_png")
eval_im_format = ("png", "jpg")

SR_name = "128_20210515-104132_Final"
SR_path = "saved_models/SR/" + SR_name
SR_model = Model_util.load_model(SR_path)
HR, LR, bicubic = SR.predict_model(SR_model, evaluation_128_image_path)
HR.save(SR_path + "/hr.jpg")
LR.save(SR_path + "/lr.jpg")
bicubic.save(SR_path + "/bi.jpg")
# call evaluation function
evaluate_model_single_tag(
    SR_path,
    "SR",
    eval_im_128_folder,
    eval_im_format,
    eval_sample_frac=1.0, # sample size: fraction of evalutation data used
    verify_id=True
    )

#**********************************************************************************************************************

# Load DN-256 model and do prediction of one image.
DN_256_name = "256_20210515-221533_best"
DN_256_path = "saved_models/DN/" + DN_256_name
DN_256 = Model_util.load_model(DN_256_path)
denoised_256, noisy_256 = DN.predict_model(DN_256, evaluation_256_image_path)
noisy_256.save(DN_256_path + "/n256.jpg")
denoised_256.save(DN_256_path + "/dn256.jpg")

# call evaluation function
evaluate_model_single_tag(
    DN_256_path,
    "DN",
    eval_im_256_folder,
    eval_im_format,
    eval_sample_frac=1.0, # sample size: fraction of evalutation data used
    verify_id=True
    )


# Load DN-128 model and do prediction of one image.
DN_128_name = "128_20210516-152054"
DN_128_path = "saved_models/DN/" + DN_128_name
DN_128 = Model_util.load_model(DN_128_path)
denoised_128, noisy_128 = DN.predict_model(DN_128, evaluation_128_image_path)
noisy_128.save(DN_128_path + "/n128.jpg")
denoised_128.save(DN_128_path + "/dn128.jpg")

evaluate_model_single_tag(
    DN_128_path,
    "DN",
    eval_im_128_folder,
    eval_im_format,
    eval_sample_frac=1.0, # sample size: fraction of evalutation data used
    verify_id=True
    )
#**********************************************************************************************************************

# SRDN pipeline

evaluate_model_single_tag(
    SR_path,
    "SRDN",
    eval_im_256_folder,
    eval_im_format,
    eval_sample_frac=1.0, # sample size: fraction of evalutation data used
    second_model_name=DN_256_path,
    verify_id=True
    )



#**********************************************************************************************************************

# DNSR pipeline

evaluate_model_single_tag(
    DN_128_path,
    "DNSR",
    eval_im_256_folder,
    eval_im_format,
    eval_sample_frac=1.0, # sample size: fraction of evalutation data used
    second_model_name=SR_path,
    verify_id=True
    )

#**********************************************************************************************************************









