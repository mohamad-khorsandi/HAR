import os.path
import cv2
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip


def dataset_from_video(video_path, tar):
    if not os.path.exists(tar):
        raise Exception()

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 9 != 0:
            continue

        cur_img = os.path.join(tar, f'{frame_count}.jpeg')
        cv2.imwrite(cur_img, frame)


def trim_video(video_path, start_time, end_time, output_path):
    clip = VideoFileClip(video_path)
    subclip = clip.subclip(start_time, end_time)
    subclip.write_videofile(output_path, codec='libx264')
    clip.reader.close()


if __name__ == '__main__':
    # dataset_from_video('data/sleeping_dataset.mp4', 'data/pic_data2/sleeping')
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")