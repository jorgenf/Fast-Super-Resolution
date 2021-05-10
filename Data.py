from PIL import Image
import numpy as np
import os
import math

def import_SR_images(split, loc="images/funiegan/", LR=120, HR=240):
    X = []
    Y = []
    n = 0
    l = len([name for name in os.listdir(loc) if os.path.isfile(os.path.join(loc, name))])
    for filename in os.listdir(loc):
        p = math.ceil((n/l)*100)
        print("\r" + "Importing images: " + str(p) + "%\t" + "#"*p, end="")
        img = Image.open(loc + filename)
        w, h = img.size
        img = img.convert("L")
        img = img.crop((0, 0, min(w, h), min(w, h)))

        x = img.resize((LR, LR), resample=Image.BICUBIC)
        y = img.resize((HR, HR), resample=Image.BICUBIC)

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
    data = {"name": loc, "LR": LR, "HR": HR, "X_train" : X_train, "Y_train" : Y_train, "X_test" : X_test, "Y_test" : Y_test}
    return data

def import_DN_images(split, x_loc, y_loc, dim):
    X = []
    Y = []
    for loc in [x_loc,y_loc]:
        n = 0
        l = len([name for name in os.listdir(loc) if os.path.isfile(os.path.join(loc, name))])
        for filename in os.listdir(loc):
            img = Image.open(loc + filename)
            img = img.convert("L")
            img = img.crop((0, 0, dim, dim))
            img = np.asarray(img)
            p = math.ceil((n/l)*100)
            if loc == x_loc:
                print("\r" + "Importing noisy images: " + str(p) + "%\t" + "#"*p, end="")
                X.append(img)
            else:
                print("\r" + "Importing clean images: " + str(p) + "%\t" + "#" * p, end="")
                Y.append(img)
            n += 1
        print("\nImported " + str(len(X)) + " images!")


    X_train = np.asarray(X[0: l - round(l * split)])
    Y_train = np.asarray(Y[0: l - round(l * split)])
    X_test = np.asarray(X[l - round(l * split):])
    Y_test = np.asarray(Y[l - round(l * split):])
    data = {"x_loc": x_loc, "y_loc": y_loc, "X_train" : X_train, "Y_train" : Y_train, "X_test" : X_test, "Y_test" : Y_test}
    return data