"""
@Brief: Test Emotion Tracker
@Author: Ty Nguyen
@Email: tynguyen.tech@gmail.com
"""
from cascade_fer.utils import EmotionTracker
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def test_emotion_tracker():
    tracker = EmotionTracker(max_frames=10)
    np.random.seed(0)
    gt_major_emotions_list = [
        4,
        4,
        0,
        0,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        6,
        0,
    ]
    emotions_list = np.random.randint(0, 7, size=20)
    # logging.info("Emotions list: {}".format(emotions_list))
    for i, emotion in enumerate(emotions_list):
        # logging.info("Emotion: {}".format(emotion))
        major_emotion = tracker.update(emotion)
        # logging.info("Major emotion: {}".format(major_emotion))
        # logging.info("Emotion tracker: {}".format(tracker))
        assert major_emotion == gt_major_emotions_list[i]
    print(gt_major_emotions_list)


if __name__ == "__main__":
    test_emotion_tracker()
