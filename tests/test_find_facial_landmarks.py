# import the necessary packages
import cv2
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

from cascade_fer.models.cv_face_detector import CVFaceDetector


def test_single_image():
    cv_detector = CVFaceDetector()
    image = cv2.imread("tests/data/sample_face.png")
    _, faces = cv_detector.detect_faces(image)
    # Show the image
    cv_detector.cv_draw_face_landmarks(image, faces)
    plt.imshow(image[:, :, ::-1])
    plt.show()


def test_live_stream():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        logging.info("Could not open camera! Stopping")
        video.release()
        return
    video_recorder = cv2.VideoWriter(
        "results/face_detection.mp4", cv2.VideoWriter_fourcc(*"XVID"), 30, (640, 480),
    )
    cv_detector = CVFaceDetector()
    while True:
        ret, frame = video.read()
        if not ret:
            logging.info("Could not read frame! Stopping")
            break
        _, faces = cv_detector.detect_faces(frame)
        # Show the image
        cv_detector.cv_draw_face_landmarks(frame, faces)
        video_recorder.write(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    video_recorder.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test_single_image()
    test_live_stream()
