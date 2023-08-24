from action_picture import ActionPicture
import cv2
import joblib
from sklearn.metrics import accuracy_score

from dataset import Dataset
from sklearn import svm


def train():
    dataset = Dataset('data/points')
    dataset.load()
    dataset.train_test_split(test_size=.2)

    model = svm.SVC(kernel='rbf')
    model.fit(dataset.x_train, dataset.y_train)
    y_pred = model.predict(dataset.x_test)
    accuracy = accuracy_score(dataset.y_test, y_pred)
    print("Accuracy:", accuracy)
    joblib.dump(model, f'svm{accuracy}')


def video_inference(model_filename, video_path=None):
    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    dataset = Dataset('data/points')
    dataset.load_labels()
    model = joblib.load(model_filename)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        action_pic = ActionPicture(frame)
        action_pic.extract_keypoints()
        frame = action_pic.draw_keypoints()

        pred = model.predict(action_pic.person_list[0].flatten())[0]
        print(dataset.labels[pred])
        cv2.imshow('Video Playback', frame)
        cv2.waitKey(1)


def picture_inference(model_filename, img):
    model = joblib.load(model_filename)
    dataset = Dataset('data/points')
    dataset.load_labels()

    action_pic = ActionPicture(img)
    action_pic.extract_keypoints()

    pred = model.predict(action_pic.person_list[0].flatten())[0]
    print(dataset.labels[pred])


if __name__ == '__main__':
    # train()
    dataset = Dataset('data/points')
    dataset.add_all_images_to_category('2_standing', 'data/walk or run/walk_or_run_train/train/walk')
    # video_inference('svm0.83')
    picture_inference('svm0.83', 'data/walk or run/walk_or_run_train/train/walk/walk_12f08de0.png')
