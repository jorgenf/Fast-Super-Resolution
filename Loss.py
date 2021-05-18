import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Calculates the L1 loss.
def getL1(Y_true, Y_pred):
    L1 = tf.keras.losses.mean_absolute_error(Y_true, Y_pred)
    return L1

# Calculates the SSIM loss.
def getSSIM(Y_true, Y_pred):
    return tf.reduce_mean(tf.image.ssim(Y_true, Y_pred, 1))

# Returns the combines L1 and SSIM loss.
def getL1SSIM(Y_true, Y_pred):
    return getL1(Y_true,Y_pred) + getSSIM(Y_true, Y_pred)


