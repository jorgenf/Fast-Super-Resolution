from pathlib import Path
import numpy as np
import cv2

rejected_folder = Path("evaluation_images/rejected_tags/ch1_fading_x_png")
valid_folder = Path("evaluation_images/valid_tags/ch1_fading_x_png")

def find_stats(img_path):
    img = cv2.imread(img_path, 0)
    return img.mean(), img.std()


rejected_imgs = rejected_folder.glob("*.png")
valid_imgs = valid_folder.glob("*.png")

reject_mean = 0
reject_std = 0
n = 0
for img in rejected_imgs:
    n += 1
    mean, std = find_stats(str(img))
    reject_mean += mean
    reject_std += std

print(f"Rejects:\nMean: {reject_mean / n :.2f}\tStd: {reject_std / n :.2f}")

print("---")
for img in valid_imgs:
    find_stats(str(img))