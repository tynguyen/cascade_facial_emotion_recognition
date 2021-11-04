from typing import Deque
import requests
import os
from collections import deque
from enum import Enum


# 7 types of emotions
class Emotions(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SUPPRISE = 5
    NEUTRAL = 6


# 2 groups of emotions
class EmotionGroups(Enum):
    NEGATIVE = -1
    POSITIVE = 1
    NEUTRAL = 0


# Map emotions to emotion groups
def emotion_to_group(emotion: Emotions) -> EmotionGroups:
    if emotion in [Emotions.ANGRY, Emotions.DISGUST, Emotions.FEAR, Emotions.SAD]:
        return EmotionGroups.NEGATIVE
    elif emotion == Emotions.HAPPY or emotion == Emotions.SUPPRISE:
        return EmotionGroups.POSITIVE
    else:
        return EmotionGroups.NEUTRAL


def download_file_from_url(save_path: str, url: str):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_path):
        r = requests.get(url, allow_redirects=True)
        open(save_path, "wb").write(r.content)


class EmotionTracker(object):
    """ A simple tracker that can be used to keep track of emotion detection over frames
    Args:
        object ([type]): [description]
    """

    def __init__(self, max_frames: int = 1) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.emotions = deque(maxlen=max_frames)
        self.emotion_groups = deque(maxlen=max_frames)

    def add_emotion(self, emotion_idx: int) -> None:
        """Add an emotion to the tracker

        Args:
            emotion_idx (int): The emotion index
        """
        self.emotions.append(emotion_idx)
        self.emotion_groups.append(emotion_to_group(Emotions(emotion_idx)).value)

    def get_major_emotion(self) -> int:
        """Get the most frequent emotion in the tracker

        Returns:
            int: The emotion index
        """
        return max(set(self.emotions), key=self.emotions.count)

    def update(self, emotion_idx: int) -> int:
        """Update the tracker with a new emotion and return the most frequent emotion

        Args:
            emotion_idx (int): The emotion index
        """
        self.add_emotion(emotion_idx)

    def get_major_emotion_group(self) -> int:
        """Get the most frequent emotion group in the tracker

        Returns:
            int: The emotion group index
        """
        return max(set(self.emotion_groups), key=self.emotion_groups.count)

    def __len__(self) -> int:
        return len(self.emotions)

    def __str__(self) -> str:
        return super().__str__() + ": " + str(self.emotions)
