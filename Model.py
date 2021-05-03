import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Data
import datetime
from PIL import Image
import time

'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Enable/disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
'''

def create_model(dim, n, d, s, m):
    # Input layer. Takes the image shape and one channel for greyscale images.
    inputs = keras.Input(shape=(dim,dim,1,))

    # The convolutional layers of the model. Keeps size of feature maps constant with same padding.

    # Feature extraction
    f_1 = 5
    n_1 = d
    conv56_5 = layers.Conv2D(filters=n_1, kernel_size=(f_1,f_1), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1) ,activation=keras.activations.relu)
    x = conv56_5(inputs)

    # Shrinking
    f_2 = 1
    n_2 = s
    conv12_1 = layers.Conv2D(filters=n_2, kernel_size=(f_2,f_2), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1), activation=keras.activations.relu)
    x = conv12_1(x)

    # Non-linear mapping
    f_3 = 3
    n_3 = s
    for l in range(0,m):
        conv56_1 = layers.Conv2D(filters=n_3, kernel_size=(f_3,f_3), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1), activation=keras.activations.relu)
        x = conv56_1(x)

    # Expanding
    f_4 = 1
    n_4 = d
    conv12_1 = layers.Conv2D(filters=n_4, kernel_size=(f_4,f_4), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1), activation=keras.activations.relu)
    x = conv12_1(x)

    # Deconvolution
    f_5 = 9
    n_5 = 1
    deconv1_9 = layers.Conv2DTranspose(filters=1, kernel_size=(f_5,f_5), strides=(n,n), padding="same", kernel_initializer=tf.initializers.random_normal(0.1), activation=keras.activations.relu)
    outputs = deconv1_9(x)

    # Creates the model by assigning the input and output layer.
    model = keras.Model(inputs=inputs, outputs=outputs, name="FSRCNN")
    # Gives model information. Second line outputs diagram.
    model.summary()
    #keras.utils.plot_model(model, "my_first_model.png")

    # Compiles model with selected features.
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    return model

def train_model(model, data, epochs, batch_size, save_path="saved_models", model_alias=None):
    if model_alias is None:
        model_alias = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Normalize data
    X_train = data["X_train"] / 255
    Y_train = data["Y_train"] / 255
    X_test = data["X_test"] / 255
    Y_test = data["Y_test"] / 255

    # infer image dimensions by size of array
    dim = X_train[0].shape[0]

    # Trains the model.
    log_dir = "logs/fit/" + model_alias
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

    # Saved model in separate directory
    # dtmin = str(datetime.datetime.now().minute)
    # dth = str(datetime.datetime.now().hour)
    # dtd = str(datetime.datetime.now().day)
    # dtm = str(datetime.datetime.now().month)
    # dir = working_dir + "/saved_models/" + dim + "_" + dth + "-" + dtmin + "_" + dtd + "-" + dtm
    dir = f"{save_path}/{dim}_{model_alias}"
    try:
        os.mkdir(dir)
    except:
        print(f"Failed to create directory \"{dir}\"")
    model.save(dir + "/")

    # Saves info about model
    with open(dir + "/assets/summary.txt", 'w') as info:
        info.write("Training dataset: " + data["name"] + "\tsize: " + str(len(data["X_train"]) + len(data["X_test"])) + "\n")
        info.write("Epochs: " + str(epochs) + "\tBatch size: " + str(batch_size) + "\n")
        model.summary(print_fn=lambda x: info.write(x + '\n'))
        info.close()

    # os.mkdir(dir + "/assets/images")
    # dim = tf.shape(X_train)[-1]
    # img0, LR0, b0 = predict_model(model, working_dir + "/images/CH1_frames/charuco_36-18-0.jpg", dim=dim)
    # img0.save(dir + "/assets/images/charuco_36-18-0.jpg")
    # img1, LR1, b1 = predict_model(model, working_dir + "/images/CH1_frames/charuco_CH1_35-15-21.jpg", dim=dim)
    # img1.save(dir + "/assets/images/charuco_CH1_35-15-21.jpg")
    # img2, LR2, b2 = predict_model(model, working_dir + "/images/CH1_frames/charuco_CH1_35-15-30.jpg", dim=dim)
    # img2.save(dir + "/assets/images/charuco_CH1_35-15-30.jpg")
    # img3, LR3, b3 = predict_model(model, working_dir + "/images/CH1_frames/charuco_CH1_35-15-98.jpg", dim=dim)
    # img3.save(dir + "/assets/images/charuco_CH1_35-15-98.jpg")
    # img4, LR4, b4 = predict_model(model, working_dir + "/images/CH1_frames/charuco_CH1_35-15-4.jpg", dim=dim)
    # img4.save(dir + "/assets/images/charuco_CH1_35-15-4.jpg")
    # img5, LR5, b5 = predict_model(model, working_dir + "/images/FunieGanData/nm_0up.jpg", dim=dim)
    # img5.save(dir + "/assets/images/nm_0up.jpg")
    # img6, LR6, b6 = predict_model(model, working_dir + "/images/FunieGanData/nm_78up.jpg", dim=dim)
    # img6.save(dir + "/assets/images/nm_78up.jpg")
    # img7, LR7, b7 = predict_model(model, working_dir + "/images/FunieGanData/nm_76up.jpg", dim=dim)
    # img7.save(dir + "/assets/images/nm_76up.jpg")
    # img8, LR8, b8 = predict_model(model, working_dir + "/images/FunieGanData/nm_286up.jpg", dim=dim)
    # img8.save(dir + "/assets/images/nm_286up.jpg")
    # img9, LR9, b9 = predict_model(model, working_dir + "/images/FunieGanData/nm_255up.jpg", dim=dim)
    # img9.save(dir + "/assets/images/nm_255up.jpg")

    return model


def load_model(dir):
    return keras.models.load_model(dir)


def predict_model(model, image_dir, input_dim, magnification=2):

    LR = Image.open(image_dir)
    LR = LR.convert("L")
    w, h = LR.size
    LR = LR.crop((0, 0, min(w, h), min(w, h)))
    LR = LR.resize((input_dim, input_dim), resample=Image.BICUBIC)

    x = tf.keras.preprocessing.image.img_to_array(LR)
    x = x / 255
    x = tf.reshape(x, (1, input_dim, input_dim,))
    start = time.time()
    y = model.predict(x)
    stop = time.time()
    # print("Elapsed time: " + str(stop-start))
    y = tf.reshape(y, (input_dim*magnification, input_dim*magnification, 1)) * 255
    HR = tf.keras.preprocessing.image.array_to_img(y)
    bicubic = LR.resize((input_dim*magnification,input_dim*magnification), resample=Image.ANTIALIAS)
    
    return HR, LR, bicubic

def prelu(x, i):
    return keras.activations.relu()


if __name__ == "__main__":
    n = 2
    d = 56
    s = 12
    m = 0
    dim = 100
    # dataset = "FunieGanData"
    dataset = "CH1_frames"
    data = Data.import_images(loc="training_images/" + dataset + "/",split = 0.1, LR=dim, HR=dim*2)

    model = create_model(dim, n, d, s, m)
    model = train_model(model, data, epochs=10, batch_size=32, model_alias="tester")
