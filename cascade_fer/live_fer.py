"""
@Brief: Live facial emotion recognition
@Author: Ty Nguyen
@Email: tynguyen.tech@gmail.com
"""
### General imports ###
from __future__ import division

import numpy as np
import cv2
import os
import logging
from scipy.linalg.matfuncs import fractional_matrix_power

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

global shape_x
global shape_y
global input_shape
global nClasses


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


def stream(num_accumulated_frames=150):
    """
    @Brief: Stream video from webcam and detect faces
    @Param: num_accumulated_frames: number of frames to accumulate to determine the emotion. 1: no accumulation
    """
    shape_x = 48
    shape_y = 48

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    face_landmarks_detector = CVFaceDetector()
    fer_model = prepare_fer_model()

    video_capture = cv2.VideoCapture(0)
    # Emotion tracker
    emotion_tracker = EmotionTracker(max_frames=num_accumulated_frames)

    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()
        frame, rects = face_landmarks_detector.detect_faces(frame)
        frame_copy = frame.copy()
        for (i, rect) in enumerate(rects):
            shape = face_landmarks_detector.detect_landmarks(frame, rect)

            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # face = gray[y : y + h, x : x + w]
            face = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
            # Zoom on extracted face
            face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))

            # Cast type float
            face = face.astype(np.float32)

            # Scale
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))

            # Make Prediction
            prediction = fer_model.predict(face)
            prediction_result = np.argmax(prediction)

            # Update the emotion tracker
            emotion_tracker.update(prediction_result)
            major_emotion = emotion_tracker.get_major_emotion()
            major_emotion_group = emotion_tracker.get_major_emotion_group()

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

            for (j, k) in shape:
                cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)
            # A copy of the frame for displaying the majority emotion
            frame_copy = frame.copy()

            # 1. Add prediction probabilities
            cv2.putText(
                frame,
                "----------------",
                (40, 100 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                0,
            )
            cv2.putText(
                frame,
                "Emotional report : Face #" + str(i + 1),
                (40, 120 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                0,
            )
            cv2.putText(
                frame,
                "Angry : " + str(round(prediction[0][0], 3)),
                (40, 140 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                0,
            )
            cv2.putText(
                frame,
                "Disgust : " + str(round(prediction[0][1], 3)),
                (40, 160 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                0,
            )
            cv2.putText(
                frame,
                "Fear : " + str(round(prediction[0][2], 3)),
                (40, 180 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Happy : " + str(round(prediction[0][3], 3)),
                (40, 200 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Sad : " + str(round(prediction[0][4], 3)),
                (40, 220 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Surprise : " + str(round(prediction[0][5], 3)),
                (40, 240 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Neutral : " + str(round(prediction[0][6], 3)),
                (40, 260 + 180 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )

            # 2. Annotate main image with a label
            cv2.putText(
                frame,
                Emotions(prediction_result).name,
                (x + w - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display the majority emotion
            cv2.putText(
                frame_copy,
                f"Major emotion over {num_accumulated_frames} frames: ",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # Emotion
            cv2.putText(
                frame_copy,
                Emotions(major_emotion).name,
                (x + w - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # Emotion group
            cv2.putText(
                frame_copy,
                EmotionGroups(major_emotion_group).name,
                (x + w - 10, y - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

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

        cv2.putText(
            frame,
            "Number of Faces : " + str(len(rects)),
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            155,
            1,
        )
        cv2.imshow("Video", np.hstack([frame, frame_copy]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    stream()


if __name__ == "__main__":
    main()
