import os
from action_picture import ActionPicture, Person
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.labels = {}

        self._x = None
        self._y = None

    def load(self):
        i = 0
        self._x = []
        self._y = []
        cat_size = min([len(os.listdir(os.path.join(self.dataset_path, cat_dir))) for cat_dir in os.listdir(self.dataset_path)])

        for cat in os.listdir(self.dataset_path):
            cat_num = int(cat.split('_')[0])
            cat_path = os.path.join(self.dataset_path, cat)

            keypoints = os.listdir(cat_path)
            for i in range(cat_size):
                person = Person.from_keypoint_path(os.path.join(cat_path, keypoints[i]))
                features = person.preprocess()
                self._x.append(features)
                self._y.append(cat_num)
            i += 1

        self._x = np.array(self._x)
        self._y = np.array(self._y)

    def train_test_split(self, test_size=0.2, shuffle=True):
        # x_train, x_test, y_train, y_test
        return train_test_split(self._x, self._y, test_size=test_size, shuffle=shuffle)

    def add_all_images_to_category(self, cat_name, image_dir):
        cat_path = os.path.join(self.dataset_path, cat_name)
        if not os.path.exists(cat_path):
            raise Exception('category does not exist')

        for img_name in os.listdir(image_dir):
            action_pic = ActionPicture.from_path(os.path.join(image_dir, img_name))
            action_pic.yolo_inference()
            tar_path = os.path.join(self.dataset_path, cat_name, img_name)
            action_pic.save_keypoints(tar_path)

    def load_labels(self):
        for cat in os.listdir(self.dataset_path):
            cat_num = int(cat.split('_')[0])
            cat_name = cat.split('_')[1]
            self.labels[cat_num] = cat_name

    def get_label_list(self):
        if len(self.labels) == 0:
            self.load_labels()

        return [self.labels[i] for i in range(len(self.labels))]
