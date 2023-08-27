import cv2
from ultralytics import YOLO
import copy
from person import Person

yolo_model = YOLO('models/yolov8s-pose.pt')


class ActionPicture:
    def __init__(self, img):
        self._img = img
        self.person_list = []

    @classmethod
    def from_path(cls, img_path):
        return ActionPicture(cv2.imread(img_path))

    def yolo_inference(self):
        res_list = yolo_model(self._img)[0]
        for res in res_list:
            person = Person.from_yolo_res(res)
            self.person_list.append(person)

    def save_keypoints(self, tar_path):
        if not self.person_list:
            self.yolo_inference()

        for i, person in enumerate(self.person_list):
            person.save_keypoints(tar_path + f'_p{i}')

    def draw_keypoints(self):
        if not self.person_list:
            self.yolo_inference()

        for person in self.person_list:
            for point in person.keypoints:
                point = (int(round(point[0])), int(round(point[1])))
                self._img = cv2.circle(self._img, point, 2, (0, 0, 255), -1)

        return self._img

    def draw_rectangle(self):
        if not self.person_list:
            self.yolo_inference()
        for person in self.person_list:
            self._img = cv2.rectangle(self._img, person.box_start_point, person.box_end_point, (255, 0, 0), 1)

        return self._img


