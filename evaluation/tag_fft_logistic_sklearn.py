from pathlib import Path
# import pickle
import joblib
import time

import numpy as np
from numpy.lib.histograms import histogram
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm

# from create_data.create_tag_dataset import create_dataset
from create_data.create_fft_dataset import create_dataset

def train_model(valid, rejects, dims):

    # create dataset
    x_data, y_label = create_dataset(valid, rejects, dims=dims)

    # split into test and train data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_label,
        test_size=0.3
        # random_state=1
    )

    # create model
    # clf = LogisticRegression(
    clf = svm.SVC(
        kernel="poly"
        # verbose=1
        # random_state=1
        ).fit(x_train, y_train)
    return clf, clf.score(x_test, y_test)

def test_parameters(n, dim_range, valid, rejects):
    from tqdm import trange
    # n = 1000
    for dims in dim_range:
        # score = [train_and_score(dims) for x in range(n)]
        bin_total_score = 0
        # for i in trange(n):
        for i in range(n):
            model, score = train_model(valid, rejects, dims)
            bin_total_score += score
        print(f"Dimensions: {dims}\t Acc: {bin_total_score/n:.5f}")


if __name__ == "__main__":

    """ dataset """
    # create dataset
    # valid = "evaluation_images/valid_tags/charuco_CH1_35-15_x_png"
    # rejects = "evaluation_images/rejected_tags/charuco_CH1_35-15_sorted"
    valid = "evaluation_images/valid_tags/combined4"
    rejects = "evaluation_images/rejected_tags/test4"

    # test the model performance for different parameters
    # test_dims = [(4, 4), (8, 8), (16, 16,), (32, 32)]
    # test_parameters(10, test_dims, valid, rejects)

    # bins in histogram
    dims = (16, 16)

    # save model to path
    save_path = Path("evaluation/logistic_models")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # create and save model
    model, test_score = train_model(valid, rejects, dims)
    joblib.dump(model, f"{save_path}/{model_name}.joblib")
    print(f"Accuracy: {test_score}")
    print(f"Saved to path \"{save_path}/{model_name}\"")