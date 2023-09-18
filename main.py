import configparser
import copy
import os.path

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import config
from action_recognizer import ActionRecognizer
from dataset import Dataset
from model import Model
from nn_model import NNModel


def train_svm(model_dir):
    dataset = Dataset(config.dataset_path)
    dataset.load()
    x_train, x_test, y_train, y_test = dataset.train_test_split()

    model = Model.for_train()
    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test, dataset.get_label_list())

    model.save(model_dir)


def train_network(model_dir):
    dataset = Dataset(config.dataset_path)
    dataset.load()
    x_train, x_test, y_train, y_test = dataset.train_test_split()

    model = NNModel.for_train()
    model.fit(x_train, y_train, epochs=20)
    model.evaluate(x_test, y_test, dataset.get_label_list())

    model.save(model_dir)


def test_model(model_filename):
    dataset = Dataset(config.dataset_path)
    dataset.load()

    x_train, y_train, x_test, y_test = dataset.train_test_split(.9)
    model = Model.for_inference(model_filename)
    model.evaluate(x_test, y_test, dataset.get_label_list())


def record_video(video_path, frame_count):
    cap = cv2.VideoCapture(video_path)
    out = ActionRecognizer.config_writer(cap)

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)


if __name__ == '__main__':
    # train_svm('models')
    # dataset = Dataset(config.dataset_path)
    # dataset.add_all_images_to_category('0_sleeping', 'data/pic_data2/sleeping')
    # video_path = "data/bb.mp4"
    # video_path = "rtsp://admin:nimda110@192.168.10.56:554/cam/realmonitor?channel=3&subtype=0"
    video_path = "data/g_workout.mkv"
    action_recognizer = ActionRecognizer()
    action_recognizer.video_inference(video_path, False)
    # record_video(video_path, 1000)
    # train_network('models')
