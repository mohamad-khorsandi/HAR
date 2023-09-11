import cv2
import numpy as np
from motrackers import IOUTracker
from motrackers.utils import draw_tracks
from person_history import PersonHistory
import config
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
            # tracker_output_format="visdrone_challenge",
        )

    def video_inference(self, video_path=None, write=False):
        cap = None
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)

        frame_count = 0
        out = None
        if write:
            out = self.config_writer(cap)

        while True:
            ret, frame = cap.read()
            frame_count += 1
            if not ret:
                break

            if frame_count % config.playback_frame_skip != 0:
                continue

            self.picture_inference(frame)
            cv2.imshow(self._window_name, frame)
            if write:
                out.write(frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        # out.release()
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
            frame = person.draw(frame, True, False, self._dataset.labels[person.action])

        PersonHistory.update_boxs_and_frame(tracked_objects)
        PersonHistory.update_action_and_keypoint(action_pic.person_list)

    def config_writer(self, src_cap):
        fps = int(src_cap.get(cv2.CAP_PROP_FPS))
        width = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        return cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))
