import configparser
import copy
import os.path
from moviepy.video.io.VideoFileClip import VideoFileClip
import config
from action_picture import ActionPicture
import cv2
from dataset import Dataset
from sklearn import svm
import numpy as np
from model import Model
from motrackers import IOUTracker
from motrackers.utils import draw_tracks
from action_recognizer import ActionRecognizer


def train(model_dir):
    dataset = Dataset(config.dataset_path)
    dataset.load()
    x_train, x_test, y_train, y_test = dataset.train_test_split()

    model = Model(svm.SVC(kernel='rbf'))
    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test, dataset.get_label_list())

    model.save(model_dir)


def test_model(model_filename):
    dataset = Dataset(utils)
    dataset.load()

    x_train, y_train, x_test, y_test = dataset.train_test_split(.9)
    model = Model.from_file(model_filename)
    model.evaluate(x_test, y_test, dataset.get_label_list())


if __name__ == '__main__':
    # train('models')

    # test_model('models/svm0.78')

    # dataset = Dataset('data/points')
    # dataset.add_all_images_to_category('2_standing', 'data/walk or run/walk_or_run_train/train/walk')

    video_path = "rtsp://admin:nimda110@192.168.10.56:554/cam/realmonitor?channel=3&subtype=0"
    # video_path = "rtsp://admin:admin123@192.168.10.68:554/cam/realmonitor?channel=1&subtype=0"
    # if not os.path.exists(video_path):
    #     trim_video('data/namaz.mp4', 60.0, 400.0, video_path)

    action_recognizer = ActionRecognizer()
    action_recognizer.video_inference(video_path)

    # picture_inference('svm0.83', 'data/walk or run/walk_or_run_train/train/walk/walk_12f08de0.png')
