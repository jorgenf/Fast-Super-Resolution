from PIL import Image
import numpy as np
import os


def import_images(split, loc="images/"):
    X = []
    Y = []
    n = 0
    l = len([name for name in os.listdir(loc) if os.path.isfile(os.path.join(loc, name))])
    for filename in os.listdir(loc):
        p = round((n/l)*100)
        print("\r" + "Importing images: " + str(p) + "%\t" + "#"*p, end="")
        img = Image.open(loc + filename)
        img = img.convert("L")

        x = img.resize((160, 120), resample=Image.BICUBIC)
        y = img.resize((320, 240), resample=Image.BICUBIC)

        x = x.crop((0, 0, 120, 120))
        y = y.crop((0, 0, 240, 240))

        x = np.asarray(x)
        y = np.asarray(y)

        X.append(x)
        Y.append(y)
        n += 1

    print("\nImported " + str(len(X)) + " images!")

    X_train = np.asarray(X[0: l - round(l * split)])
    Y_train = np.asarray(Y[0: l - round(l * split)])
    X_test = np.asarray(X[l - round(l * split):])
    Y_test = np.asarray(Y[l - round(l * split):])
    data = {"X_train" : X_train, "Y_train" : Y_train, "X_test" : X_test, "Y_test" : Y_test}
    return data

def get_image(u):
    u = Image.fromarray(u)
    Image._show(u)

#import_images(0.1)