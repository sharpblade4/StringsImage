#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from scipy import interpolate
from PIL import Image

RADIUS = 50  # cm
SCREWS_AMOUNT = 470
STRING_LENGTH = 5.5 * 100 * 1000  # cm


class Gui:
    def __init__(self, screws_position: List[Tuple[float, float]]):
        self._screws_position = screws_position

    def draw_strings(self, strings: List[List[int]]) -> None:
        for line in strings:
            self._draw_screws_string_connection(line[0], line[1])

    def draw_screws(self) -> None:
        plt.scatter(self._screws_position[:, 0], self._screws_position[:, 1], s=1)

    def _draw_screws_string_connection(self, screw1: int, screw2: int) -> None:
        pos_1 = self._screws_position[screw1]
        pos_2 = self._screws_position[screw2]
        plt.plot([pos_1[0], pos_2[0]], [pos_1[1], pos_2[1]],
                 alpha=0.17, c='black', linewidth=0.5)

    def show(self) -> None:
        plt.axis('off')
        plt.gcf().tight_layout()
        plt.gcf().gca().set_aspect('equal', 'datalim')
        plt.show()


class Engine:
    def __init__(self):
        self._leftover_string = STRING_LENGTH
        self._screws_position = None
        self.init_screws()
        # TODO Engine should handle all image processing (blurring, quantisizing)

    def init_screws(self) -> None:
        x0, y0 = (0, 0)
        rads = (2 * np.pi / SCREWS_AMOUNT) * np.arange(SCREWS_AMOUNT)
        xys = np.array([[x0 + RADIUS * np.sin(angle), y0 + RADIUS * np.cos(angle)]
                        for angle in rads])
        self._screws_position = xys

    def get_screws_positions(self) -> List[Tuple[float, float]]:
        assert self._screws_position is not None, "screws not initialized"
        return self._screws_position.copy()

    @staticmethod
    def just_try(randomize: bool) -> List[List[int]]:
        if randomize:
            amount = 500
            return np.random.randint(SCREWS_AMOUNT, size=(amount, 2))
        else:
            jump = 70
            return np.vstack([np.arange(SCREWS_AMOUNT - jump),
                              np.arange(SCREWS_AMOUNT - jump) + jump]).T

    def calculate_string_usage(self, screw1: int, screw2: int) -> float:
        if not (0 <= screw1 < SCREWS_AMOUNT) or \
                not (0 <= screw2 < SCREWS_AMOUNT):
            raise IndexError
        p1 = np.array(self._screws_position[screw1])
        p2 = np.array(self._screws_position[screw2])
        euclidean_distance = np.sqrt(np.sum(np.power(p2 - p1, 2)))
        return euclidean_distance

    def use_string(self, amount: float) -> None:
        if self._leftover_string - amount < 0:
            raise Exception("Used too much string")
        self._leftover_string -= amount


class Algo:  # TODO separate all classes to different files.
    def __init__(self, image_path, screws_position):
        self._screws_position = screws_position
        self._curr_state = None
        self._grid = None
        self._prepare_image(image_path)
        self._init_grid()

    def _init_grid(self):
        assert self._curr_state is not None, "must load and prepare image before using grid"
        im_height, im_width = self._curr_state.shape  # TODO assuming RGB
        self._grid = interpolate.interp2d(np.arange(im_width), np.arange(im_height),
                                          self._curr_state)

    def _prepare_image(self, image_path: str):
        # load image as grayscale
        im = np.array(Image.open(image_path).convert('L'))
        # crop image to circle center
        center = np.floor(0.5 * np.array(im.shape)).astype(np.int32)
        im_radius = min(center)
        im_crop = im[center[0] - im_radius: center[0] + im_radius,
                  center[1] - im_radius: center[1] + im_radius]
        # scale to circle size
        im = np.array(Image.fromarray(im_crop).resize((RADIUS * 2, RADIUS * 2)))
        self._curr_state = im.copy()

    def get_next(self, current_screw: int):  # TODO check: state : Dict, -> int:
        if not (0 <= current_screw < SCREWS_AMOUNT):
            raise IndexError
        best_score = self._score_line(current_screw, current_screw + 1)
        for i in range(SCREWS_AMOUNT):
            if i in (current_screw, current_screw + 1):
                continue
            

        # TODO implement (cont)

    def _score_line(self, screw1: int, screw2: int) -> float:
        p1, p2 = [np.array(self._screws_position[i]) for i in (screw1, screw2)]
        euclidean_distance = np.sqrt(np.sum(np.power(p2 - p1, 2)))
        xs = p1[0] + (p2[0] - p1[0]) * np.arange(np.ceil(euclidean_distance)) / euclidean_distance
        ys = p1[1] + (p2[1] - p1[1]) * np.arange(np.ceil(euclidean_distance)) / euclidean_distance
        score = 0
        for point in zip(xs, ys):
            intensity = self._grid(*point)
            score -= intensity
            # TODO plan: after choosing line, increasing state's intensity by 1 (recreate grid), including increasing whites to 256 etc
        return score


if __name__ == '__main__':
    engine = Engine()
    screws_pos = engine.get_screws_positions()
    ui = Gui(screws_pos)

    tux_path = '/home/ru/Pictures/tux-100677393-large.jpg'
    algo = Algo(tux_path, screws_pos)

    trial = engine.just_try(randomize=True)
    ui.draw_screws()
    ui.draw_strings(trial)
    ui.show()
