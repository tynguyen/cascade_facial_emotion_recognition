"""
@Brief: Test Live facial emotion recognition
@Author: Ty Nguyen
@Email: tynguyen.tech@gmail.com
"""
from genericpath import isdir
from cascade_fer import live_fer
import logging

logging.basicConfig(level=logging.INFO)


def test_live_fer():
    """
    Test Live facial emotion recognition
    """
    logging.info("Testing Live facial emotion recognition")
    live_fer.stream(
        num_accumulated_frames=60,
        video_path="results/emotion_recognition.avi",
        is_display_raw_pred=True,
    )


if __name__ == "__main__":
    test_live_fer()
