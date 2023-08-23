import cv2
import numpy as np
from ultralytics import YOLO
import copy


class ActionPicture:
    def __init__(self, img_path):
        self.img_path = img_path
        self._img = cv2.imread(img_path)
        self.label = None
        self._person_list = []
        self.no_person = False

    def extract_keypoints(self):
        model = YOLO('models/yolov8n-pose.pt')

        res = model(self._img)

        for person in res:
            self._person_list.append(Person(person.keypoints.xy.numpy()[0]))

    def save_keypoints(self, tar_path):
        np.save(tar_path, self.keypoints)

    def draw_keypoints(self):
        if not self._person_list:
            self.extract_keypoints()
        pointy_img = copy.deepcopy(self._img)

        for person in self._person_list:
            for point in person.keypoints:
                point = (int(round(point[0])), int(round(point[1])))
                pointy_img = cv2.circle(pointy_img, point, 2, (0, 0, 255), -1)

        cv2.imshow('pointy', pointy_img)
        cv2.waitKey(10000000)


class Person:
    def __init__(self, keypoints):
        self.keypoints = keypoints
