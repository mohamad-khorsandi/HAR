import numpy as np

import config
from dataset import Dataset


class PersonHistory:
    def __init__(self, id):
        self.box_xywh_history = []
        self.frame = []

        self.action_history = []
        self.keypoint_history = []

        self.id = id
        self.partially_updated = False

    person_history_dict = {}

    @classmethod
    def update_boxs_and_frame(cls, tracked_objects):
        for tracked_object in tracked_objects:
            id = tracked_object[1]
            frame_number = tracked_object[0]

            person_hist = cls.get_or_create(id)

            person_hist.frame.append(frame_number)
            person_hist.box_xywh_history.append(np.array(tracked_object[2: 6], dtype=float))
            person_hist.partially_updated = True

    @classmethod
    def update_action_and_keypoint(cls, new_person_list):
        for new_person in new_person_list:
            for _, person_history in cls.person_history_dict.items():
                if np.all(new_person.box_xywh == person_history.get_last_xywh()) and person_history.partially_updated:
                    person_history.action_history.append(new_person.action)
                    person_history.keypoint_history.append(new_person.keypoints)
                    person_history.partially_updated = False
                    break

    @classmethod
    def report(cls):
        for _, person_hist in cls.person_history_dict.items():
            print(person_hist)

    def get_last_xywh(self):
        if len(self.box_xywh_history) == 0:
            raise Exception()
        return self.box_xywh_history[len(self.box_xywh_history) - 1]

    @classmethod
    def get_or_create(cls, id):
        person_hist = cls.person_history_dict.get(id)

        if person_hist is None:
            person_hist = PersonHistory(id)
            cls.person_history_dict[id] = person_hist

        return person_hist

    def __str__(self):
        if len(self.action_history) == 0:
            raise Exception

        dataset = Dataset.for_inference(config.dataset_path)
        total = len(self.action_history)
        rate0 = round(100 * self.action_history.count(0) / total)
        rate1 = round(100 * self.action_history.count(1) / total)
        rate2 = round(100 * self.action_history.count(2) / total)

        return (f'{self.id}: {dataset.id_to_label(0)}={rate0}, '
                f'{dataset.id_to_label(1)}={rate1}, '
                f'{dataset.id_to_label(2)}={rate2}')
