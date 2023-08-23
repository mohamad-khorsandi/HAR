from action_picture import ActionPicture
import cv2
import joblib
from sklearn.metrics import accuracy_score

from dataset import Dataset
from sklearn import svm


def main():
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
    model = joblib.load(model_filename)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        action_pic = ActionPicture(frame)
        kepoints = action_pic.extract_keypoints()[0]

        frame = draw_keypoints(kepoints, frame)
        kepoints = kepoints.reshape(1, 34)
        print(model.predict(kepoints))
        cv2.imshow('Video Playback', frame)
        cv2.waitKey(1)





if __name__ == '__main__':
    dataset = Dataset('data/points')
    dataset.add_all_images_to_category('standing', 'data/black_guy_dataset/standing')
