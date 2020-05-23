import dlib
import os
import cv2
from .eye import Eye


class Tracker(object):
    def __init__(self):
        self.frame = None
        self.right_eye = None
        self.left_eye = None

        # load dlib deps
        self._detector = dlib.get_frontal_face_detector()
        cwd = os.path.abspath(os.path.dirname(__file__))
        model = os.path.abspath(
            os.path.join(cwd, "models/shape_predictor_68_face_landmarks.dat")
        )
        self._predictor = dlib.shape_predictor(model)

    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    @staticmethod
    def middle(p1, p2):
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _analyze(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector(frame)
        try:
            landmarks = self._predictor(frame, faces[0])
            self.right_eye = Eye(frame, landmarks, 0)
            self.left_eye = Eye(frame, landmarks, 1)
        except (ZeroDivisionError, IndexError):
            self.right_eye = None
            self.left_eye = None

    def get_frame(self):
        frame = self.frame.copy()
        color = (0, 0, 255)  # RED (in BGR)
        x_left, y_left = self.left_eye.coords
        x_left = int(x_left + self.left_eye.origin[0])
        y_left = int(y_left + self.left_eye.origin[1])
        x_right, y_right = self.right_eye.coords
        x_right = int(x_right + self.right_eye.origin[0])
        y_right = int(y_right + self.right_eye.origin[1])
        cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
        cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
        cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
        cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

    @property
    def eyes_detected(self):
        return self.right_eye != None and self.left_eye != None

    @property
    def gaze(self):
        right_coords = self.right_eye.coords
        left_coords = self.left_eye.coords
        return (
            (left_coords[0] + right_coords[0]) / 2,
            (left_coords[1] + right_coords[1]) / 2,
        )
