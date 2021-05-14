from pathlib import Path
# import pickle
import joblib
import time

import numpy as np
from numpy.lib.histograms import histogram
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm

from create_data.create_tag_dataset import create_dataset

def train_model(valid, rejects, hist_bins):

    # create dataset
    x_data, y_label = create_dataset(valid, rejects, bins=hist_bins)

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

def test_parameters(n, bin_range, valid, rejects):
    from tqdm import trange
    # n = 1000
    for bins in bin_range:
        # score = [train_and_score(bins) for x in range(n)]
        bin_total_score = 0
        # for i in trange(n):
        for i in range(n):
            model, score = train_model(valid, rejects, bins)
            bin_total_score += score
        print(f"Bins: {bins:02d}\t Acc: {bin_total_score/n:.5f}")


if __name__ == "__main__":

    """ dataset """
    # create dataset
    # valid = "evaluation_images/valid_tags/charuco_CH1_35-15_x_png"
    # rejects = "evaluation_images/rejected_tags/charuco_CH1_35-15_sorted"
    valid = "evaluation_images/valid_tags/combined"
    rejects = "evaluation_images/rejected_tags/combined"

    # test the model performance for different parameters
    # test_parameters(10, ([1, 2, 4, 8, 16, 32, 64]), valid, rejects)

    # bins in histogram
    histogram_bins = 8

    # save model to path
    save_path = Path("evaluation/logistic_models")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # create and save model
    model, test_score = train_model(valid, rejects, histogram_bins)
    joblib.dump(model, f"{save_path}/{model_name}.joblib")
    print(f"Accuracy: {test_score}")
    print(f"Saved to path \"{save_path}/{model_name}\"")