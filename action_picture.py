import cv2
from ultralytics import YOLO
import utils
from person import Person
import config


class ActionPicture:
    def __init__(self, img):
        self._img = img
        self.person_list = []
        self.bounding_box = None
        self.acc_scores = []

    _yolo_model = YOLO(config.pose_estimation_model)

    @classmethod
    def from_path(cls, img_path):
        return ActionPicture(cv2.imread(img_path))

    def yolo_inference(self):
        res_list = self._yolo_model(self._img)[0]
        self.bounding_box = res_list.boxes.xywh.numpy()
        self.acc_scores = res_list.boxes.conf.numpy()
        for res in res_list:
            person = Person.for_inference(res)
            self.person_list.append(person)
