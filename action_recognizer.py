import cv2
import numpy as np
from motrackers import IOUTracker
from motrackers.utils import draw_tracks
from person_history import PersonHistory
import config
import utils
from action_picture import ActionPicture
from dataset import Dataset


class ActionRecognizer:
    def __init__(self):
        self._dataset = Dataset.for_inference(config.dataset_path)

        self._window_name = config.window_name
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 1500, 2300)

        self._tracker = IOUTracker(
            max_lost=300,
            iou_threshold=0.3,
            min_detection_confidence=0.5,
            # max_detection_confidence=0.7,
            tracker_output_format="visdrone_challenge",
        )

    def video_inference(self, video_path=None):
        cap = None
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            frame_count += 1
            if frame_count % 2 != 0 or not ret:
                continue

            self.picture_inference(frame)
            cv2.imshow(self._window_name, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        PersonHistory.report()

    def picture_inference(self, frame):
        action_pic = ActionPicture(frame)
        action_pic.yolo_inference()

        tracked_objects = self._tracker.update(action_pic.bounding_box, action_pic.acc_scores,
                                               np.zeros_like(action_pic.acc_scores))
        frame = draw_tracks(frame, tracked_objects)

        for person in action_pic.person_list:
            person.predict_action()
            frame = person.draw(frame, False, True, self._dataset.labels[person.action])

        PersonHistory.update_boxs_and_frame(tracked_objects)
        PersonHistory.update_action_and_keypoint(action_pic.person_list)
