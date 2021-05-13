import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from PIL import Image
import time
import Data
import SR
import math
from Loss import getL1SSIM, getL1, getSSIM

def create_model(dim, loss="L1SSIM"):

    input = keras.Input(shape=(dim, dim, 1,))
    x = input
    skip_values = []
    # Convolution layers
    for i in range(8):
        num_filters = min(64 * pow(2, i), 512)
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=4, strides=2, padding="SAME")
        x = conv(x)
        if i != 0:
            batch_norm = keras.layers.BatchNormalization()
            x = batch_norm(x)
        relu = keras.layers.ReLU()
        x = relu(x)
        if i < 7:
            skip_values.append(x)

    # Deconvolution layers
    num_filters = 512
    for i in range(7):
        deconv = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=4, strides=2, padding="SAME")
        x = deconv(x)
        batch_norm_deconv = keras.layers.BatchNormalization()
        x = batch_norm_deconv(x)
        relu = keras.layers.ReLU()
        x = relu(x)
        if i < 3:
            dropout = keras.layers.Dropout(rate=0.5)
            x = dropout(x)
        else:
            num_filters /= 2
        concat = keras.layers.Concatenate(axis=3)
        x = concat([x,skip_values[6-i]])

    output_layer = keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation=keras.activations.tanh, padding="SAME")
    output = output_layer(x)
    model = keras.Model(inputs=input, outputs=output)
    if loss == "L1SSIM":
        model.compile(loss=getL1SSIM, optimizer=keras.optimizers.Adam(), metrics=[getL1SSIM, getL1, getSSIM, "mean_squared_error"])
    else:
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(),
                      metrics=["mean_squared_error"])
    model.summary()
    return model

def train_model(model, data, epochs, batch_size, directory=".", model_alias=None):
    if model_alias is None:
        model_alias = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Normalize data
    X_train = data["X_train"] / 255
    Y_train = data["Y_train"] / 255
    X_test = data["X_test"] / 255
    Y_test = data["Y_test"] / 255

    # infer image dimensions by size of array
    dim = model.layers[0].get_input_at(0).get_shape().as_list()[1]

    # Trains the model.
    log_dir = directory + "logs/fit/" + model_alias
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[tensorboard_callback])

    # Evaluates model with test data.
    test_scores = model.evaluate(X_test, Y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    dir = f"{directory}/saved_models/DN/{dim}_{model_alias}"
    try:
        os.mkdir(dir)
    except:
        print(f"Failed to create directory \"{dir}\"")
    model.save(dir + "/")

    # Saves info about model
    with open(dir + "/assets/summary.txt", 'w') as info:
        info.write("Training dataset: " + data["x_loc"] + "\tsize: " + str(len(data["X_train"]) + len(data["X_test"])) + "\n")
        info.write("Epochs: " + str(epochs) + "\tBatch size: " + str(batch_size) + "\n")
        model.summary(print_fn=lambda x: info.write(x + '\n'))
        info.close()

    os.mkdir(dir + "/assets/images")
    img0, LR0 = predict_model(model, directory + "/training_images/CH1_frames/charuco_36-18-0.jpg")
    img0.save(dir + "/assets/images/charuco_36-18-0.jpg")
    img1, LR1 = predict_model(model, directory + "/training_images/CH1_frames/charuco_CH1_35-15-21.jpg")
    img1.save(dir + "/assets/images/charuco_CH1_35-15-21.jpg")
    img2, LR2 = predict_model(model, directory + "/training_images/CH1_frames/charuco_CH1_35-15-30.jpg")
    img2.save(dir + "/assets/images/charuco_CH1_35-15-30.jpg")
    img3, LR3 = predict_model(model, directory + "/training_images/CH1_frames/charuco_CH1_35-15-98.jpg")
    img3.save(dir + "/assets/images/charuco_CH1_35-15-98.jpg")
    img4, LR4 = predict_model(model, directory + "/training_images/CH1_frames/charuco_CH1_35-15-4.jpg")
    img4.save(dir + "/assets/images/charuco_CH1_35-15-4.jpg")
    img5, LR5 = predict_model(model, directory + "/training_images/FunieGanData/nm_0up.jpg")
    img5.save(dir + "/assets/images/nm_0up.jpg")
    img6, LR6 = predict_model(model, directory + "/training_images/FunieGanData/nm_78up.jpg")
    img6.save(dir + "/assets/images/nm_78up.jpg")
    img7, LR7 = predict_model(model, directory + "/training_images/FunieGanData/nm_76up.jpg")
    img7.save(dir + "/assets/images/nm_76up.jpg")
    img8, LR8 = predict_model(model, directory + "/training_images/FunieGanData/nm_286up.jpg")
    img8.save(dir + "/assets/images/nm_286up.jpg")
    img9, LR9 = predict_model(model, directory + "/training_images/FunieGanData/nm_255up.jpg")
    img9.save(dir + "/assets/images/nm_255up.jpg")
    return model

def load_model(dir):
    return keras.models.load_model(dir)

def predict_model(model, image_dir):
    input_dim = model.layers[0].get_input_at(0).get_shape().as_list()[1]
    noisy = Image.open(image_dir)
    noisy = noisy.convert("L")
    noisy = noisy.crop((0, 0, input_dim, input_dim))

    x = tf.keras.preprocessing.image.img_to_array(noisy)
    x = x / 255
    x = tf.reshape(x, (1, input_dim, input_dim,))
    start = time.time()
    y = model.predict(x)
    stop = time.time()
    # print("Elapsed time: " + str(stop-start))
    y = tf.reshape(y, (input_dim, input_dim, 1)) * 255
    denoised = tf.keras.preprocessing.image.array_to_img(y)

    return denoised, noisy
