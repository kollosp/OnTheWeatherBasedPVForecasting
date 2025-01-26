from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from .Plotter import Plotter, PlotterObject

import logging
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Plotterv2(Plotter):
    """
    Class for interactive plotting. It handle mouse movement and click. After data selection it executes callback.
    """
    def __init__(self, x_axis:List | np.ndarray, list_of_data_or_plotter_object:List[np.ndarray | PlotterObject],
                 save_callback: Callable[[np.ndarray], None] | None = None,
                 list_of_line_names: List[str] | None = None,
                 displayed_window_size:int = 1_000,
                 one_time_jump: int = 300):
        super().__init__(
            x_axis = x_axis,
            list_of_data_or_plotter_object = list_of_data_or_plotter_object,
            displayed_window_size = displayed_window_size,
            list_of_line_names = list_of_line_names,
            one_time_jump = one_time_jump)
        self._x_clicked = []
        self._current_mouse_x = 0
        self._rects = []

        self._save_callback = save_callback
        self._annotation_line = None


    def show(self) -> None:
        super().show()

        # mouse events
        # self._fig.canvas.mpl_connect('key_release_event', self.on_release) #repaint on release
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_click(self, event):
        super().on_click(event)

        value = None
        sign = ord(event.key[0])
        if event.key == "escape":
            logger.info("Escape clicked. Clearing marked regions.")
            self._x_clicked = []
        elif ord('0') <= sign <= ord('9'):
             value = sign - ord('0') # classes from 0-9
        elif ord('A') <= sign <= ord('Z'):
            value = sign - ord('A') + 10 # classes from 10 - ...
        elif sign == ord('c'):
            if self._save_callback is not None:
                logger.info(
                    f"I'm going to save annotations. Classes used = {np.unique(self._annotation_data)}. Current ts position is:"
                    f"{int(100 * self.current_position / len(self.x_axis))}%")

                self._save_callback(self._annotation_data)
                logger.info(f"Done.")
            else:
                logger.info(f"Cannot save annotations. The save function not passed in constructor")

        if len(self._x_clicked) % 2 == 0 and value is not None:
            logger.info(f"Assignment made. Assign class {value}")
            logger.info("    Clearing marked regions.")
            self.annotate_data(int(value))
            self._x_clicked = []
            self.remove_rectangles()
            self.repaint()

    def draw_rectangles(self):
        ax = self.ax[0]
        for i in range(1, len(self._x_clicked), 2):
            p0 = self._x_clicked[i-1]
            p1 = self._x_clicked[i]

            rect = ax.axvspan(p0, p1, alpha=0.5, color='red')
            self._rects.append(rect)

    def remove_rectangles(self):
        for rect in self._rects:
            rect.remove()
        self._rects = []

    def draw_current_rectangle(self):
        #check if rect already exists
        rects_count = int(len(self._x_clicked) / 2)
        if len(self._rects) > rects_count:
            self._rects[-1].remove()
            self._rects = self._rects[:-1]

        ax = self.ax[0]
        if len(self._x_clicked) % 2 == 1:
            rect = ax.axvspan(self._x_clicked[-1], self._current_mouse_x, alpha=0.5, color='red')
            self._rects.append(rect)

    def annotate_data(self, annotation_class):
        x = self._x_axis
        for i in range(1, len(self._x_clicked), 2):
            p0 = self._x_clicked[i-1]
            p1 = self._x_clicked[i]
            mx = max(p0,p1)
            mi = min(p0,p1)
            self._annotation_data[np.all([(x <= mx), (x >= mi)], axis=0)] = annotation_class


    def repaint(self) -> None:
        y = self.get_window(self._annotation_data)
        x = self.get_window(self.x_axis)
        if self._annotation_line is None:
            self._annotation_line, = self.ax[0].plot(x, y)
        else:
            self._annotation_line.set_data(x,y)

        super().repaint()

    def on_mouse_click(self, event):
        self._x_clicked.append(event.xdata)
        logger.info(f"Mouse clicked on X-coordinate={event.xdata}. Current regions number={int(len(self._x_clicked)/2)}")

        if len(self._x_clicked) % 2 == 0:
            self.remove_rectangles()
            self.draw_rectangles()
        else:
            self.draw_current_rectangle()

        self.repaint()

    def on_mouse_move(self, event):
        if event.xdata is not None:
            if abs(self._current_mouse_x - event.xdata) > 1_000:
                self._current_mouse_x = event.xdata
                self.draw_current_rectangle()
                self.repaint()


