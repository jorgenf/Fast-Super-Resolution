import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from PIL import Image
import time


def create_model(dim, n, d, s, m):
    # Input layer. Takes the image shape and one channel for greyscale images.
    inputs = keras.Input(shape=(dim,dim,1,))

    # The convolutional layers of the model. Keeps size of feature maps constant with same padding.

    # Feature extraction
    f_1 = 5
    n_1 = d
    conv56_5 = layers.Conv2D(filters=n_1, kernel_size=(f_1,f_1), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1))
    x = conv56_5(inputs)
    prelu1 = layers.PReLU(alpha_initializer=tf.random_normal_initializer(0.1))
    x = prelu1(x)

    # Shrinking
    f_2 = 1
    n_2 = s
    conv12_1 = layers.Conv2D(filters=n_2, kernel_size=(f_2,f_2), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1))
    x = conv12_1(x)
    prelu2 = layers.PReLU(alpha_initializer=tf.random_normal_initializer(0.1))
    x = prelu2(x)


    # Non-linear mapping
    f_3 = 3
    n_3 = s
    for l in range(0,m):
        conv56_1 = layers.Conv2D(filters=n_3, kernel_size=(f_3,f_3), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1))
        x = conv56_1(x)
        prelu3 = layers.PReLU(alpha_initializer=tf.random_normal_initializer(0.1))
        x = prelu3(x)

    # Expanding
    f_4 = 1
    n_4 = d
    conv12_1 = layers.Conv2D(filters=n_4, kernel_size=(f_4,f_4), strides=(1,1), padding="same", kernel_initializer=tf.initializers.random_normal(0.1))
    x = conv12_1(x)
    prelu4 = layers.PReLU(alpha_initializer=tf.random_normal_initializer(0.1))
    x = prelu4(x)

    # Deconvolution
    f_5 = 9
    n_5 = 1
    deconv1_9 = layers.Conv2DTranspose(filters=1, kernel_size=(f_5,f_5), strides=(n,n), padding="same", kernel_initializer=tf.initializers.random_normal(0.1))
    x = deconv1_9(x)
    prelu5 = layers.PReLU(alpha_initializer=tf.random_normal_initializer(0.1))
    outputs = prelu5(x)

    # Creates the model by assigning the input and output layer.
    model = keras.Model(inputs=inputs, outputs=outputs, name="FSRCNN")
    # Gives model information. Second line outputs diagram.
    model.summary()
    #keras.utils.plot_model(model, "my_first_model.png")

    # Compiles model with selected features.
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), metrics=["mean_squared_error"])

    return model

def train_model(model, data, epochs, batch_size, working_dir):
    # Normalize data

    X_train = data["X_train"] / 255
    Y_train = data["Y_train"] / 255
    X_test = data["X_test"] / 255
    Y_test = data["Y_test"] / 255

    # Trains the model.
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    dtmin = str(datetime.datetime.now().minute)
    dth = str(datetime.datetime.now().hour)
    dtd = str(datetime.datetime.now().day)
    dtm = str(datetime.datetime.now().month)
    dir = working_dir + "/saved_models/" + dth + "-" + dtmin + "_" + dtd + "-" + dtm
    os.mkdir(dir)
    model.save(dir + "/")

    # Saves info about model
    with open(dir + "/assets/summary.txt", 'w') as info:
        info.write("Training dataset: " + data["name"] + "\tsize: " + str(len(data["X_train"]) + len(data["X_test"])) + "\n")
        info.write("Epochs: " + str(epochs) + "\tBatch size: " + str(batch_size) + "\n")
        model.summary(print_fn=lambda x: info.write(x + '\n'))
        info.close()

    os.mkdir(dir + "/assets/images")
    dim = tf.shape(X_train)[-1]
    img0, LR0, b0 = predict_model(model, working_dir + "/images/charuco/charuco_36-18-0.jpg", dim=dim)
    img0.save(dir + "/assets/images/charuco_36-18-0.jpg")
    img1, LR1, b1 = predict_model(model, working_dir + "/images/charuco/charuco_CH1_35-15-21.jpg", dim=dim)
    img1.save(dir + "/assets/images/charuco_CH1_35-15-21.jpg")
    img2, LR2, b2 = predict_model(model, working_dir + "/images/charuco/charuco_CH1_35-15-30.jpg", dim=dim)
    img2.save(dir + "/assets/images/charuco_CH1_35-15-30.jpg")
    img3, LR3, b3 = predict_model(model, working_dir + "/images/charuco/charuco_CH1_35-15-98.jpg", dim=dim)
    img3.save(dir + "/assets/images/charuco_CH1_35-15-98.jpg")
    img4, LR4, b4 = predict_model(model, working_dir + "/images/charuco/charuco_CH1_35-15-4.jpg", dim=dim)
    img4.save(dir + "/assets/images/charuco_CH1_35-15-4.jpg")
    img5, LR5, b5 = predict_model(model, working_dir + "/images/funiegan/nm_0up.jpg", dim=dim)
    img5.save(dir + "/assets/images/nm_0up.jpg")
    img6, LR6, b6 = predict_model(model, working_dir + "/images/funiegan/nm_78up.jpg", dim=dim)
    img6.save(dir + "/assets/images/nm_78up.jpg")
    img7, LR7, b7 = predict_model(model, working_dir + "/images/funiegan/nm_76up.jpg", dim=dim)
    img7.save(dir + "/assets/images/nm_76up.jpg")
    img8, LR8, b8 = predict_model(model, working_dir + "/images/funiegan/nm_286up.jpg", dim=dim)
    img8.save(dir + "/assets/images/nm_286up.jpg")
    img9, LR9, b9 = predict_model(model, working_dir + "/images/funiegan/nm_255up.jpg", dim=dim)
    img9.save(dir + "/assets/images/nm_255up.jpg")

    return model


def load_model(dir):
    return keras.models.load_model(dir)


def predict_model(model, image_dir, dim):

    LR = Image.open(image_dir)
    LR = LR.convert("L")
    w, h = LR.size
    LR = LR.crop((0, 0, min(w, h), min(w, h)))
    LR = LR.resize((dim, dim), resample=Image.BICUBIC)

    x = tf.keras.preprocessing.image.img_to_array(LR)
    x = x / 255
    x = tf.reshape(x, (1, dim, dim,))
    start = time.time()
    y = model.predict(x)
    stop = time.time()
    print("Elapsed time: " + str(stop-start))
    y = tf.reshape(y, (dim*2, dim*2, 1)) * 255
    HR = tf.keras.preprocessing.image.array_to_img(y)
    bicubic = LR.resize((dim*2,dim*2), resample=Image.ANTIALIAS)
    return HR, LR, bicubic






