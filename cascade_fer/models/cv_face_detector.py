"""
@Brief: Cascade face detector using OpenCV
@Author: Ty Nguyen
@Source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
@Email: tynguyen.tech@gmail.com
"""
import os
import cascade_fer
from cascade_fer.utils import download_file_from_url
import numpy as np
import imutils
from imutils import face_utils
import dlib
import cv2
import logging

logging.basicConfig(level=logging.INFO)


class CVFaceDetector(object):
    def __init__(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()

        # Download the shape predictor file if it does not exist
        shape_predictor_file = os.path.join(
            os.path.dirname(cascade_fer.__file__),
            "ckpts/LandmarksDetector/shape_predictor_68_face_landmarks.dat",
        )
        url = "https://docs.google.com/uc?export=download&id=1O0YZAGG7-UbkrlOkBe41pKQ7GccPlyx2"
        if not os.path.exists(shape_predictor_file):
            logging.info("Downloading facial landmark model from {}".format(url))
            download_file_from_url(shape_predictor_file, url)

        self.predictor = dlib.shape_predictor(shape_predictor_file)

    def detect_faces(self, image, max_img_width=None, max_img_height=None):
        """
        Detects faces in an image and returns the bounding boxes.
        :param image: The image to detect faces in.
        :param max_img_width: The maximum width of the input image.
        :param max_img_height: The maximum height of the input image.
        """
        # load the input image, resize it, and convert it to grayscale
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise TypeError("image must be a string or numpy array")
        if max_img_width:
            image = imutils.resize(image, width=max_img_width)
        if max_img_height:
            image = imutils.resize(image, height=max_img_height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        faces = self.detector(gray, 1)
        # Return the detected faces and resized image
        return image, faces

    def detect_landmarks(self, gray_img, face):
        """Detect landmarks of a face

        Args:
            gray_img (numpy.ndarray): Grayscale image
            face ([_dlib_pybind11.rectangle]): the face detected by dlib
        Returns:
            np.array (68 x ...): the landmarks of the face
        shape = self.predictor(face, face.rect)
        shape = face_utils.shape_to_np(shape)
        return shape
        """
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        elif len(gray_img.shape) == 2:
            pass
        else:
            raise TypeError("gray_img must be a grayscale image")

        shape = self.predictor(gray_img, face)
        shape = face_utils.shape_to_np(shape)
        return shape

    def cv_draw_face_landmarks(self, image, faces):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # loop over the face detections
        for (i, rect) in enumerate(faces):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(
                image,
                "Face #{}".format(i + 1),
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
