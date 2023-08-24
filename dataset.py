import os
from action_picture import ActionPicture, Person
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.labels = {}

        self.x = []
        self.y = []

        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

    def load(self):
        i = 0
        for cat in os.listdir(self.dataset_path):
            cat_num = int(cat.split('_')[0])
            cat_path = os.path.join(self.dataset_path, cat)

            for keypoints_file in os.listdir(cat_path):
                person = Person.load_person(os.path.join(cat_path, keypoints_file))
                self.x.append(person.flatten())
                self.y.append(cat_num)
            i += 1

    def train_test_split(self, test_size=0.2, shuffle=True):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, shuffle=shuffle)
        self.x_train = np.array(x_train).reshape(-1,34)
        self.x_test = np.array(x_test).reshape(-1, 34)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

    def add_all_images_to_category(self, cat_name, image_dir):
        cat_path = os.path.join(self.dataset_path, cat_name)
        if not os.path.exists(cat_path):
            raise Exception('category does not exist')

        for img_name in os.listdir(image_dir):
            action_pic = ActionPicture.from_path(os.path.join(image_dir, img_name))
            action_pic.extract_keypoints()
            tar_path = os.path.join(self.dataset_path, cat_name, img_name)
            action_pic.save_keypoints(tar_path)

    def load_labels(self):
        for cat in os.listdir(self.dataset_path):
            cat_num = int(cat.split('_')[0])
            cat_name = cat.split('_')[1]
            self.labels[cat_num] = cat_name