"""
@Brief: Live facial emotion recognition
@Author: Ty Nguyen
@Email: tynguyen.tech@gmail.com
"""
### General imports ###
from __future__ import division

import numpy as np
import cv2
import logging

from cascade_fer import CascadeFER

logging.basicConfig(level=logging.INFO)


def stream(
    num_accumulated_frames=150,
    video_path="/tmp/emotion_recognition.mp4",
    is_display_raw_pred=False,
):
    """
    @Brief: Stream video from webcam and detect faces
    @Param: num_accumulated_frames: number of frames to accumulate to determine the emotion. 1: no accumulation
    @Param: is_display_raw_pred: display raw prediction or not. True: display raw prediction
    """
    video_recorder = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (640, 480),
    )

    video_capture = cv2.VideoCapture(0)
    # Emotion detector and tracker
    fer_tracker = CascadeFER(window_duration_in_frames=num_accumulated_frames)

    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()
        fer_tracker(frame)
        # Display the emotion detected
        fer_tracker.display_emotion(frame, is_display_raw_pred=is_display_raw_pred)
        # Display the detected face
        fer_tracker.display_facial_landmarks(frame)
        cv2.imshow("Video", frame)
        video_recorder.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    video_capture.release()
    video_recorder.release()
    cv2.destroyAllWindows()


def main():
    stream()


if __name__ == "__main__":
    main()
