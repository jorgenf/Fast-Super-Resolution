import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from PIL import Image
import time
import Model
from Loss import getL1SSIM, getL1, getSSIM

def create_model(dim, activation="relu", loss="L1SSIM"):

    input = keras.Input(shape=(dim, dim, 1,))
    x = input
    skip_values = []
    # Convolution layers
    n_conv = 8 if dim == 256 else 7
    for i in range(n_conv):
        num_filters = min(64 * pow(2, i), 512)
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=4, strides=2, padding="SAME")
        x = conv(x)
        if i != 0:
            batch_norm = keras.layers.BatchNormalization()
            x = batch_norm(x)
        if activation == "relu":
            relu = keras.layers.ReLU()
            x = relu(x)
        elif activation == "lrelu":
            lrelu = keras.layers.LeakyReLU(alpha=0.3)
            x = lrelu(x)
        else:
            raise Exception("No valid activation function chosen.")
        if i < n_conv - 1:
            skip_values.append(x)

    # Deconvolution layers
    num_filters = 512
    n_deconv = n_conv - 1
    for i in range(n_deconv):
        deconv = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=4, strides=2, padding="SAME")
        x = deconv(x)
        batch_norm_deconv = keras.layers.BatchNormalization()
        x = batch_norm_deconv(x)
        if activation == "relu":
            relu = keras.layers.ReLU()
            x = relu(x)
        elif activation == "lrelu":
            lrelu = keras.layers.LeakyReLU(alpha=0.3)
            x = lrelu(x)
        else:
            raise Exception("No valid activation function chosen.")
        if i < n_deconv - 4:
            dropout = keras.layers.Dropout(rate=0.5)
            x = dropout(x)
        else:
            num_filters /= 2
        concat = keras.layers.Concatenate(axis=3)
        x = concat([x,skip_values[n_deconv-1-i]])

    output_layer = keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation=keras.activations.tanh, padding="SAME")
    output = output_layer(x)

    model = keras.Model(inputs=input, outputs=output)
    if loss == "L1SSIM":
        model.compile(loss=getL1SSIM, optimizer=keras.optimizers.Adam(), metrics=[getL1SSIM, getL1, getSSIM, "mean_squared_error"])
    elif loss == "MSE":
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(),
                      metrics=["mean_squared_error"])
    else:
        raise Exception("No valid loss function chosen.")
    model.summary()
    return model

def train_model(model, data, epochs, batch_size, directory=".", model_alias=None):
    if model_alias is None:
        model_alias = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Normalize data
    X_train, Y_train, X_test, Y_test = Model.normalize_data(data)

    # Get input dimensions
    dim = Model.get_model_dimension(model)

    # Creates Directory
    dir, log_dir = Model.create_model_directory(directory, "DN", dim, model_alias)

    # Trains the model.
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

    model.save(dir + "/")

    # Saves info about model
    Model.save_model_info(dir, data, model, epochs, batch_size)
    model_name = f"{dim}_{model_alias}"
    return model, model_name

def predict_model(model, image_dir):
    input_dim = Model.get_model_dimension(model)
    if isinstance(image_dir, str):
        noisy = Image.open(image_dir)
    else:
        noisy = image_dir
    noisy = noisy.convert("L")
    noisy = noisy.crop((0, 0, input_dim, input_dim))
    x = tf.keras.preprocessing.image.img_to_array(noisy)
    x = x / 255
    x = tf.reshape(x, (1, input_dim, input_dim,))
    y = model.predict(x)
    y = tf.reshape(y, (input_dim, input_dim, 1)) * 255
    denoised = tf.keras.preprocessing.image.array_to_img(y)

    return denoised, noisy
