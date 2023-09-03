import configparser
import os.path
import shutil
from enum import Enum

import cv2
import numpy as np
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from ultralytics import YOLO


def show_with_matplotlib(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


def trim_video(video_path, start_time, end_time, output_path):
    clip = VideoFileClip(video_path)
    subclip = clip.subclip(start_time, end_time)
    subclip.write_videofile(output_path, codec='libx264')
    clip.reader.close()