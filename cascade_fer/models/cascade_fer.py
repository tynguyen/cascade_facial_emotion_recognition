"""
@Brief: facial emotion recognition from raw input image
@Author: Ty Nguyen
@Email: tynguyen.tech@gmail.com
"""
### General imports ###
from __future__ import division

import numpy as np
import cv2
import os
import logging

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance

from tensorflow.keras.models import load_model
from imutils import face_utils
import cascade_fer
from cascade_fer.models.cv_face_detector import CVFaceDetector
from cascade_fer.utils import (
    download_file_from_url,
    EmotionTracker,
    Emotions,
    EmotionGroups,
)

logging.basicConfig(level=logging.INFO)

# Global variables
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

(eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def prepare_fer_model():
    model_path = os.path.join(
        os.path.dirname(cascade_fer.__file__), "ckpts/EmotionXception/video.h5"
    )
    model_path_url = "https://docs.google.com/uc?export=download&id=1UZi4nOcYHrORZSPPFPOzHm4ZvSf2ejZQ"
    if not os.path.exists(model_path):
        logging.info("Downloading FER model from {}".format(model_path_url))
        download_file_from_url(model_path, model_path_url)
        logging.info("Downloaded FER model to {}".format(model_path))
    logging.info("Loading FER model from {}".format(model_path))
    return load_model(model_path)


class CascadeFER(object):
    def __init__(
        self, window_duration_in_frames=150, patch_width=48, patch_height=48,
    ):
        """
        A cascade-based emotion recognition model. It first detects faces in the image, then extracts those faces and predicts the emotion of the face.
        After that, a sliding window is used to track the dominant emotion over the last `window_duration_in_frames` frames.

        @Args:
            window_duration_in_frames (int, optional): Number of accumulated frame to determine an emotion. Defaults to 150 (~5 seconds)
            patch_width (int, optional): Width of the extracted face patch. Defaults to 48
            patch_height (int, optional): Height of the extracted face patch. Defaults to 48

        @Usage:
            >>> from cascade_fer import CascadeFER
            >>> cfer = CascadeFER()
            >>> cfer(image)
        """
        self.model = prepare_fer_model()
        self.face_landmarks_detector = CVFaceDetector()
        self.fer_model = prepare_fer_model()
        self.emotion_tracker = EmotionTracker()
        self.window_duration_in_frames = window_duration_in_frames

        # Patch size
        self.patch_width = patch_width
        self.patch_height = patch_height

        # Maintain the dominant emotion over the last `duration_in_frames` frames as well the group it belongs to
        self.dominant_emotion_index = None
        self.dominant_emotion_group = None
        self.dominant_emotion_repr = None  # dominant_emotion: {"index": emotion_index, "group": emotion_group_name, "name": emotion_name}

    def _predict(self, image):
        self._predicted_emotion_indices = []
        self._predicted_face_coords = []
        self._predicted_landmark_coords = []
        self._predicted_emotion_logits = []

        frame, rects = self.face_landmarks_detector.detect_faces(image)
        for (i, rect) in enumerate(rects):
            shape = self.face_landmarks_detector.detect_landmarks(image, rect)
            self._predicted_landmark_coords.append(shape)  # for display

            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # Append face coordinates for displaying
            self._predicted_face_coords.append([x, y, w, h])

            # face = gray[y : y + h, x : x + w]
            face = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
            # Zoom on extracted face
            face = zoom(
                face,
                (self.patch_width / face.shape[0], self.patch_height / face.shape[1]),
            )

            # Cast type float
            face = face.astype(np.float32)

            # Scale
            face /= float(face.max())
            face = np.reshape(
                face.flatten(), (1, self.patch_height, self.patch_width, 1)
            )

            # Make Prediction
            emotion_logits = self.fer_model.predict(face)
            emotion_index = np.argmax(emotion_logits)
            self._predicted_emotion_indices.append(emotion_index)
            self._predicted_emotion_logits.append(emotion_logits)

    def _update(self):
        # Update the emotion tracker
        for emotion_index in self._predicted_emotion_indices:
            self.emotion_tracker.update(emotion_index)
        # Update the dominant emotion over the last `duration_in_frames` frames
        self.dominant_emotion_index = self.emotion_tracker.get_major_emotion()
        self.dominant_emotion_group = self.emotion_tracker.get_major_emotion_group()

        dominant_emotion_name = ""
        if self.dominant_emotion_index is not None:
            dominant_emotion_name = Emotions(self.dominant_emotion_index).name
        self.dominant_emotion_repr = {
            "index": self.dominant_emotion_index,
            "group": self.dominant_emotion_group,
            "name": dominant_emotion_name,
        }

    def __call__(self, image):
        """
        Make a class instance callable
        """
        self._predict(image)
        self._update()

    def display_facial_landmarks(self, frame):
        for (i, face_coords) in enumerate(self._predicted_face_coords):
            x, y, w, h = face_coords
            # Rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                "Face #{}".format(i + 1),
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        for shape in self._predicted_landmark_coords:
            for (j, k) in shape:
                cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)

            # 3. Eye Detection and Blink Count
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compute Eye Aspect Ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # And plot its contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # 4. Detect Nose
            nose = shape[nStart:nEnd]
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

            # 5. Detect Mouth
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # 6. Detect Jaw
            jaw = shape[jStart:jEnd]
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)

            # 7. Detect Eyebrows
            ebr = shape[ebrStart:ebrEnd]
            ebrHull = cv2.convexHull(ebr)
            cv2.drawContours(frame, [ebrHull], -1, (0, 255, 0), 1)
            ebl = shape[eblStart:eblEnd]
            eblHull = cv2.convexHull(ebl)
            cv2.drawContours(frame, [eblHull], -1, (0, 255, 0), 1)

    def display_emotion(self, frame, is_display_raw_pred=False):
        if is_display_raw_pred:
            for (i, emotion_index) in enumerate(self._predicted_emotion_indices):
                x, y, w, h = self._predicted_face_coords[i]
                # display the current emotion on the frame
                cv2.putText(
                    frame,
                    Emotions(emotion_index).name,
                    (x + w - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Display the raw emotion prediction
                for j, emotion in enumerate(Emotions):
                    cv2.putText(
                        frame,
                        f"{[emotion.value]} {emotion.name}: "
                        + str(round(self._predicted_emotion_logits[i].flatten()[j], 3)),
                        (x + w - 10, y + j * 15 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        155,
                        0,
                    )

        if self.dominant_emotion_index is not None:
            # Display the dominant emotion
            cv2.putText(
                frame,
                f"Dominant emotion over {self.window_duration_in_frames} frames: ",
                (40, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 155, 0),
                2,
            )
            # Emotion
            cv2.putText(
                frame,
                f"Type: {Emotions(self.dominant_emotion_index).name}",
                (40, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 155, 0),
                2,
            )
            # Emotion group
            cv2.putText(
                frame,
                f"Group: {EmotionGroups(self.dominant_emotion_group).name}",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 155, 0),
                2,
            )
