import cv2
from ultralytics import YOLO
import utils
from person import Person


class ActionPicture:
    def __init__(self, img):
        self._img = img
        self.person_list = []
        self.bounding_box = None

    yolo_model = YOLO(utils.read_config('pose_estimation_model'))

    @classmethod
    def from_path(cls, img_path):
        return ActionPicture(cv2.imread(img_path))

    def yolo_inference(self):
        res_list = self.yolo_model(self._img)[0]
        self.bounding_box = res_list.boxes.xyxy.numpy()

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
            pass

        return self._img

    def draw_rectangle(self):
        if not self.person_list:
            self.yolo_inference()

        for person in self.person_list:
            pass

        return self._img
