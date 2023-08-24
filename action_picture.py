import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
import copy
import utils


class ActionPicture:
    def __init__(self, img):
        self._img = img
        self.person_list = []
        self.no_person = False

    @classmethod
    def from_path(cls, img_path):
        return ActionPicture(cv2.imread(img_path))

    def extract_keypoints(self):
        model = YOLO('models/yolov8n-pose.pt')

        res = model(self._img)[0]

        for person in res:
            self.person_list.append(Person(person.keypoints.xy.numpy()[0]))

    def save_keypoints(self, tar_path):
        if not self.person_list:
            self.extract_keypoints()

        for i, person in enumerate(self.person_list):
            person.save_keypoints(tar_path + f'_p{i}')

    def draw_keypoints(self):
        if not self.person_list:
            self.extract_keypoints()
        pointy_img = copy.deepcopy(self._img)

        for person in self.person_list:
            for point in person.keypoints:
                point = (int(round(point[0])), int(round(point[1])))
                pointy_img = cv2.circle(pointy_img, point, 2, (0, 0, 255), -1)

        return pointy_img


class Person:
    def __init__(self, keypoints):
        self.keypoints = keypoints

    @classmethod
    def load_person(cls, path):
        keypoints = np.load(path)
        return Person(keypoints)

    def save_keypoints(self, tar_path):
        np.save(tar_path, self.keypoints)

    def flatten(self):
        return self.keypoints.reshape(1, 34)


