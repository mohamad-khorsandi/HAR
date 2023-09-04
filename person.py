import copy
import cv2
import numpy as np
import config
import tensorflow as tf
import os

class Person:
    def __init__(self, keypoints):
        self.keypoints = keypoints
        self._feature_list = None
        self.action = None
        self.box_xywh = None
        self.box_start_point = None
        self.box_end_point = None

    _action_recognition_model = tf.keras.models.load_model(config.action_recognition_model)

    @classmethod
    def for_inference(self, yolo_res):
        keypoints = yolo_res.keypoints.xy.numpy()[0]
        person = Person(keypoints)
        box_xyxy = yolo_res.boxes.xyxy[0].numpy().astype(int)
        person.box_start_point = box_xyxy[0:2]
        person.box_end_point = box_xyxy[2:4]

        person.box_xywh = yolo_res.boxes.xywh[0].numpy().astype(float)
        return person

    @classmethod
    def from_keypoint_path(cls, path):
        keypoints = np.load(path)
        return Person(keypoints)

    def save_keypoints(self, tar_path):
        np.save(tar_path, self.keypoints)

    def _flatten(self):
        return self._feature_list.reshape(34,)

    def _max_normalization(self):
        return self._feature_list / np.max(self._feature_list, axis=0)

    def _loc_normalization(self):
        return self._feature_list - np.min(self._feature_list, axis=0)

    def preprocess(self):
        #todo check inplace of flatten
        self._feature_list = copy.copy(self.keypoints)
        # self._feature_list = self._loc_normalization()
        self._feature_list = self._max_normalization()
        self._feature_list = self._flatten()
        return self._feature_list

    def predict_action(self):
        features = self.preprocess()
        features = features.reshape(1, -1)
        self.action = self._action_recognition_model.predict(features)[0].argmax()

    def draw(self, img, keypoints=False, box=False, text=None):
        if keypoints:
            img = self.draw_rectangle(img)
        if box:
            img = self.draw_keypoints(img)
        if text:
            img = self.write_action(img, text)

        return img

    def draw_rectangle(self, img):
        return cv2.rectangle(img, self.box_start_point, self.box_end_point, (255, 0, 0), 1)

    def draw_keypoints(self, img):
        for point in self.keypoints:
            point = (int(round(point[0])), int(round(point[1])))
            img = cv2.circle(img, point, 2, (0, 0, 255), -1)
        return img

    def write_action(self, img, text):
        return cv2.putText(img, text, self.box_start_point, cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (255, 0, 0), 2, cv2.LINE_AA)
