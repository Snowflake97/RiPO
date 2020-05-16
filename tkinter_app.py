import tkinter as tk
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import time


class EyeApp:
    def __init__(self, master):
        self.master = master
        master.title("RiPO")

        self.label = tk.Label(master, text="Eye tracing app")
        self.label.pack()

        self.trace_button = tk.Button(master, text="Start tracing", command=self.run, width=50)
        self.trace_button.pack()

        self.stop_button = tk.Button(master, text="Stop tracing", command=self.stop, state=tk.DISABLED, width=50)
        self.stop_button.pack()

        self.close_button = tk.Button(master, text="Close", command=self.quit, width=50)
        self.close_button.pack()

        self.runing = True
        self.eyes_position = []
        self.thread_kill = False
        self.thread_wait = True
        self.thread = Thread(target=self.trace_eye)
        self.thread.start()

    def trace_eye(self):
        while True:
            if self.thread_kill:
                break
            if self.thread_wait:
                time.sleep(0.5)
            else:
                self.eyes_position = []
                cap = cv2.VideoCapture(0)
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

                while self.runing:
                    ret, frame = cap.read()  # read frame by frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray frame
                    faces = detector(gray)  # detect face
                    for face in faces:

                        landmarks = predictor(gray, face)  # create face landmarks

                        # coords for eyes
                        left_coords_right_eye = landmarks.part(36).x - 10, landmarks.part(37).y - 10
                        right_coords_right_eye = landmarks.part(39).x + 10, landmarks.part(41).y + 10

                        left_coords_left_eye = landmarks.part(42).x - 10, landmarks.part(43).y - 10
                        right_coords_left_eye = landmarks.part(45).x + 10, landmarks.part(47).y + 10

                        # make right and left eye roi
                        left_eye_roi = frame[left_coords_right_eye[1]: right_coords_right_eye[1],
                                       left_coords_right_eye[0]: right_coords_right_eye[0]]
                        right_eye_roi = frame[left_coords_right_eye[1]: right_coords_right_eye[1],
                                        left_coords_right_eye[0]: right_coords_right_eye[0]]

                        # make rois gray
                        gray_right_roi = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
                        gray_left_roi = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)

                        # blur
                        gray_right_roi = cv2.GaussianBlur(gray_right_roi, (3, 3), 0)
                        gray_left_roi = cv2.GaussianBlur(gray_left_roi, (3, 3), 0)

                        # thresholds
                        _, threshold_right = cv2.threshold(gray_right_roi, 60, 255, cv2.THRESH_BINARY_INV)
                        _, threshold_left = cv2.threshold(gray_left_roi, 60, 255, cv2.THRESH_BINARY_INV)

                        # grab countours
                        contours_left, hierarchy = cv2.findContours(threshold_left, cv2.RETR_TREE,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
                        contours_right, hierarchy = cv2.findContours(threshold_right, cv2.RETR_TREE,
                                                                     cv2.CHAIN_APPROX_SIMPLE)

                        # for the largest area calculate rect size and mid point
                        contours = sorted(contours_left, key=lambda x: cv2.contourArea(x), reverse=True)
                        for i, cnt in enumerate(contours):
                            (x, y, w, h) = cv2.boundingRect(cnt)
                            mid_point = (x + w / 2, y + h / 2)
                            # print(mid_point)  # print eye position (roi sizes)
                            self.eyes_position.append(mid_point)
                            x = x + left_coords_left_eye[0]
                            y = y + left_coords_left_eye[1]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # draw rect o eye
                            break

                        # for the largest area calculate rect size and mid point
                        contours = sorted(contours_right, key=lambda x: cv2.contourArea(x), reverse=True)
                        for i, cnt in enumerate(contours):
                            (x, y, w, h) = cv2.boundingRect(cnt)
                            mid_point = (x + w / 2, y + h / 2)
                            # print(mid_point)  # print eye position (roi sizes)
                            self.eyes_position.append(mid_point)
                            x = x + left_coords_right_eye[0]
                            y = y + left_coords_right_eye[1]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # draw rect o eye
                            break

                        cv2.imshow("frame", frame)  # show frame
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        # q to exit
                        break
                # quit
                cap.release()
                cv2.destroyAllWindows()
                self.thread_wait = True

    def save_results(self):

        with open("result.txt", mode="w") as f:
            for i in self.eyes_position:
                x, y = i
                f.write(f"{x}, {y}\n")
        f.close()

    def make_heatmap(self):
        results = []
        min_x = 999
        min_y = 999
        with open("result.txt", mode='r') as f:
            for line in f:
                elems = line.split(",")
                x = float(elems[0])
                y = float(elems[1])
                if x < min_x:
                    min_x = x
                if y < min_y and y > 8:
                    min_y = y
                if y > 8:
                    results.append((x, y))
        f.close()
        results = sorted(results, key=lambda x: (x[0], x[1]))
        for i, res in enumerate(results):
            x, y = res
            results[i] = (x - min_x, y - min_y)
        my_dict = {i: results.count(i) for i in results}
        if len(results)>0:
            y_size = int(sorted(results, key=lambda x: (x[1], x[0]))[-1][1])
            x_size = int(sorted(results, key=lambda x: (x[0], x[1]))[-1][0])


            array = np.zeros((y_size, x_size))

            for i in my_dict:
                value = my_dict[i]
                x, y = i
                x = int(x)
                y = int(y)
                array[y_size - 1 - y][x_size - 1 - x] = int(value)

            plt.imshow(array, cmap='hot', interpolation='nearest')
            plt.show()

    def run(self):
        self.thread_wait = False
        self.runing = True
        self.trace_button['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL

    def stop(self):
        self.runing = False
        self.thread_wait = True
        self.save_results()
        self.make_heatmap()
        self.stop_button['state'] = tk.DISABLED
        self.trace_button['state'] = tk.NORMAL

    def quit(self):
        self.runing = False
        self.thread_kill = True
        self.thread.join()
        self.master.quit()
