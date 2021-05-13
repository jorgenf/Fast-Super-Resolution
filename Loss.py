import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
import math
from scipy.ndimage import gaussian_filter
import tensorflow as tf

def getL1(Y_true, Y_pred):
    L1 = tf.keras.losses.mean_absolute_error(Y_true, Y_pred)
    return L1

def getSSIM(Y_true, Y_pred):
    return tf.reduce_mean(tf.image.ssim(Y_true, Y_pred, 1))

def getL1SSIM(Y_true, Y_pred):
    return getL1(Y_true,Y_pred) + getSSIM(Y_true, Y_pred)


