import tkinter as tk
import cv2
import dlib
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from threading import Thread, Event
import time
from gaze import Tracker


class EyeApp:
    def __init__(self, master):
        self.tracker = Tracker()

        self.master = master
        master.title("RiPO")

        self.screen_width = master.winfo_screenwidth()
        self.screen_height = master.winfo_screenheight()

        self.label = tk.Label(master, text="Eye tracing app")
        self.label.pack()

        self.calibrate_button = tk.Button(
            master, text="Run calibration", command=self.calibrate, width=50
        )
        # self.calibrate_button.pack()

        self.trace_button = tk.Button(
            master, text="Start tracing", command=self.run, state=tk.NORMAL, width=50
        )
        self.trace_button.pack()

        self.stop_button = tk.Button(
            master, text="Stop tracing", command=self.stop, state=tk.DISABLED, width=50
        )
        self.stop_button.pack()

        self.close_button = tk.Button(master, text="Close", command=self.quit, width=50)
        self.close_button.pack()

        self.running = True
        self.eyes_position = []
        self.thread_kill = False
        self.thread_wait = True
        self.capture_running = Event()
        self.capture_stopped = Event()
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
                self.capture_stopped.clear()
                self.capture_running.set()

                while self.running:
                    frame = cap.read()[1]  # read frame by frame
                    self.tracker.refresh(frame)
                    if self.tracker.eyes_detected:
                        self.eyes_position.append(self.tracker.gaze)
                        # print(self.tracker.left_eye.normalized_pos)
                        # cv2.imshow("frame", self.tracker.get_frame())
                        # cv2.waitKey(100)

                if cv2.waitKey(100) & 0xFF == ord("q"):
                    # q to exit
                    break
                cv2.destroyAllWindows()
                cap.release()
                self.capture_running.clear()
                self.capture_stopped.set()
                self.thread_wait = True

    def calibrate(self):
        self.thread_wait = False
        self.running = True
        self.capture_running.wait()
        tk.messagebox.showinfo(
            title="Calibration", message="Look at the upper left corner of the screen."
        )
        tk.messagebox.showinfo(
            title="Calibration", message="Look at the upper right corner of the screen."
        )
        tk.messagebox.showinfo(
            title="Calibration", message="Look at the lower left corner of the screen."
        )
        tk.messagebox.showinfo(
            title="Calibration", message="Look at the lower right corner of the screen."
        )
        self.running = False
        self.thread_wait = True
        self.capture_stopped.wait()
        self.generate_bounds()
        self.eyes_position = []
        self.trace_button["state"] = tk.NORMAL

    def generate_bounds(self):
        results = sorted(self.eyes_position, key=lambda x: x[0])
        self.min_x = results[0][0]
        self.max_x = results[-1][0] - self.min_x
        results = sorted(results, key=lambda x: x[1])
        self.min_y = results[0][1]
        self.max_y = results[-1][1] - self.min_y

    def save_results(self):
        with open("result.txt", mode="w") as f:
            for i in self.eyes_position:
                x, y = i
                f.write(f"{x}, {y}\n")
        f.close()

    def make_heatmap(self):
        self.generate_bounds()
        results = self.eyes_position
        # normalize and create a heatmap
        xs = []
        ys = []
        for i, elem in enumerate(results):
            elem_x = (elem[0] - self.min_x) / self.max_x
            elem_y = (elem[1] - self.min_y) / self.max_y
            x = int(elem_x * self.screen_width)
            if x < 0:
                x = 0
            elif x > self.screen_width:
                x = self.screen_width
            y = int(elem_y * self.screen_height)
            if y < 0:
                y = 0
            elif y > self.screen_height:
                y = self.screen_height
            xs.append(x)
            ys.append(y)

        plt.figure()
        ax = plt.gca()
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=10)
        extent = [self.screen_width, 0, 0, self.screen_height]
        im = plt.imshow(heatmap.T, cmap="inferno", extent=extent, origin="lower")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.show()

    def run(self):
        self.thread_wait = False
        self.running = True
        self.trace_button["state"] = tk.DISABLED
        self.stop_button["state"] = tk.NORMAL

    def stop(self):
        self.running = False
        self.thread_wait = True
        self.stop_button["state"] = tk.DISABLED
        self.trace_button["state"] = tk.NORMAL
        self.save_results()
        self.make_heatmap()
        self.eyes_position = []

    def quit(self):
        self.running = False
        self.thread_kill = True
        self.thread.join()
        self.master.quit()


if __name__ == "__main__":
    root = tk.Tk()
    my_gui = EyeApp(root)
    my_gui.make_heatmap()
