import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import Model
import Data
from PIL import Image

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Enable/disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


'''
n = 2
d = 56
s = 12
m = 0
dim = 250
dataset = "flowers"
data = Data.import_images(loc="./images/" + dataset + "/",split = 0.1, LR=dim, HR=dim*2)

model = Model.create_model(dim, n, d, s, m)
model = Model.train_model(model, data, epochs=500, batch_size=32, working_dir=".")
'''

model = Model.load_model("./saved_models/high_res")
HR, LR, b = Model.predict_model(model, "./charuco3.jpg", 250)
HR.save("./charuco_zoom.jpg")
HR.show(title="HR")
LR.show(title="LR")
b.show(title="Antialias")
'''

img = Image.open("./charuco.jpg")
img = img.crop((50,150,300,400))
img.save("./charuco3.jpg")
'''