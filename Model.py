import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Data
import datetime
from PIL import Image

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Enable/disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def create_model():
    # Input layer. Takes the image shape and one channel for greyscale images.
    inputs = keras.Input(shape=(120,120,1,))

    # The convolutional layers of the model. Keeps size of feature maps constant with same padding.
    conv56_5 = layers.Conv2D(filters=56, kernel_size=(5,5), strides=(1,1), padding="same", activation=keras.activations.relu)
    x = conv56_5(inputs)

    conv12_1 = layers.Conv2D(filters=12, kernel_size=(1,1), strides=(1,1), padding="same", activation=keras.activations.relu)
    x = conv12_1(x)

    conv12_3 = layers.Conv2D(filters=12, kernel_size=(3,3), strides=(1,1), padding="same", activation=keras.activations.relu)
    x = conv12_3(x)

    conv56_1 = layers.Conv2D(filters=56, kernel_size=(1,1), strides=(1,1), padding="same", activation=keras.activations.relu)
    x = conv56_1(x)

    # Deconvoluytion layer that scales up image.
    deconv1_9 = layers.Conv2DTranspose(filters=1, kernel_size=(9,9), strides=(1,1), dilation_rate=(15,15), activation=keras.activations.relu)
    outputs = deconv1_9(x)

    # Creates the model by assigning the input and output layer.
    model = keras.Model(inputs=inputs, outputs=outputs, name="FSRCNN")
    # Gives model information. Second line outputs diagram.
    model.summary()
    #keras.utils.plot_model(model, "my_first_model.png")

    # Compiles model with selected features.
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    return model

def train_model(model, data, epochs):
    # Normalize data

    X_train = data["X_train"] / 255
    Y_train = data["Y_train"] / 255
    X_test = data["X_test"] / 255
    Y_test = data["Y_test"] / 255

    # Trains the model.
    history = model.fit(X_train, Y_train, batch_size=64, epochs=epochs, validation_split=0.2)

    # Evaluates model with test data.
    test_scores = model.evaluate(X_test, Y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    # Saved model in separate directory
    dtmin = str(datetime.datetime.now().minute)
    dth = str(datetime.datetime.now().hour)
    dtd = str(datetime.datetime.now().day)
    dtm = str(datetime.datetime.now().month)
    dir = "./saved_models/" + dth + "-" + dtmin + "_" + dtd + "-" + dtm
    os.mkdir(dir)
    model.save(dir + "/")

def load_model(model_name):
    return keras.models.load_model("./saved_models/" + model_name)

def predict_model(model, input):

    img = Image.open("./images/" + input)
    img = img.convert("L")
    x = img.resize((160, 120), resample=Image.BICUBIC)
    x = x.crop((0, 0, 120, 120))

    x = tf.keras.preprocessing.image.img_to_array(x)
    print(tf.shape(x))
    x = x / 255

    prediction = model.predict(x)
    print(tf.shape(prediction))
    img = tf.keras.preprocessing.image.array_to_img(prediction)

    return img

data = Data.import_images(split = 0.1)
model = create_model()
train_model(model, data, 5000)

#model = load_model("13-1_27-4")
#print(model.summary())
#prediction = predict_model(model, "nm_0up.jpg")

#print(prediction)
#prediction.show()