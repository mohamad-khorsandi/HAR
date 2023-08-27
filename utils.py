import configparser
import os.path
import shutil
from enum import Enum

import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO


def create_dataset(dataset_path, cat_list):
    for action in cat_list:
        img_dir = os.path.join(dataset_path, action)
        pnt_dir = action.get_point_dir()

        for img_name in os.listdir(img_dir):
            img = cv2.imread(os.path.join(img_dir, img_name))
            points = get_keypoints(img)
            tar_path = os.path.join(pnt_dir, img_name)
            np.save(tar_path, points)


def get_keypoints(img):
    model = YOLO('yolov8n-pose.pt')

    res = model(img)
    return res[0].keypoints.xy.numpy()


def show_keypoints(img, keypoints):
    for person in keypoints:
        for point in person:
            point = (int(round(point[0])), int(round(point[1])))
            img = cv2.circle(img, point, 2, (0, 0, 255), -1)

    plt.imshow(img)
    plt.axis('off')
    plt.show()


def show_with_matplotlib(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


def read_config(key):
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.get('setting', key)
