import configparser
import copy
import os.path
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
    model.fit(x_train, y_train, epochs=35)
    model.evaluate(x_test, y_test, dataset.get_label_list())

    model.save(model_dir)


def test_model(model_filename):
    dataset = Dataset(utils)
    dataset.load()

    x_train, y_train, x_test, y_test = dataset.train_test_split(.9)
    model = Model.for_inference(model_filename)
    model.evaluate(x_test, y_test, dataset.get_label_list())


if __name__ == '__main__':
    # train('models')
    # dataset = Dataset(config.dataset2_path)
    # dataset.add_all_images_to_category('0_sleeping', 'data/pic_data2/sleeping')
    video_path = "rtsp://admin:nimda110@192.168.10.56:554/cam/realmonitor?channel=3&subtype=0"
    action_recognizer = ActionRecognizer()
    action_recognizer.video_inference(video_path)

    # train_network('models')
