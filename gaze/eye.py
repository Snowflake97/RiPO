import cv2
import numpy as np


class Eye(object):
    def __init__(self, frame, landmarks, side):
        self.frame = frame
        self.landmarks = landmarks

        if side == 0:
            self.EYE_POINTS = [36, 37, 39, 41]  # RIGHT
        else:
            self.EYE_POINTS = [42, 43, 45, 47]  # LEFT

        self.MARGIN = 10

        self._detect()

    def _detect(self):
        left_coords = (
            self.landmarks.part(self.EYE_POINTS[0]).x - self.MARGIN,
            self.landmarks.part(self.EYE_POINTS[1]).y - self.MARGIN,
        )
        right_coords = (
            self.landmarks.part(self.EYE_POINTS[2]).x + self.MARGIN,
            self.landmarks.part(self.EYE_POINTS[3]).y + self.MARGIN,
        )

        eye_roi = self.frame[
            left_coords[1] : right_coords[1], left_coords[0] : right_coords[0],
        ]

        self.origin = left_coords

        threshold = self._process(eye_roi)
        contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        moments = cv2.moments(contours[0])
        self.x = int(moments["m10"] / moments["m00"])
        self.y = int(moments["m01"] / moments["m00"])

    def _process(self, roi):
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        return cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY)[1]

    @property
    def coords(self):
        return (self.x, self.y)
