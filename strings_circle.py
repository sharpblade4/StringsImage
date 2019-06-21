#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from typing import List

RADIUS = 100  # cm
SCREWS_AMOUNT = 470
STRING_LENGTH = 5.5 * 100 * 1000  # cm


class Infrastructure:
    def __init__(self):
        self._screws_position = None

    def draw_screws(self) -> None:
        x0, y0 = (0, 0)
        rads = (2 * np.pi / SCREWS_AMOUNT) * np.arange(SCREWS_AMOUNT)
        xys = np.array([[x0 + RADIUS * np.sin(angle), y0 + RADIUS * np.cos(angle)] for angle in rads])
        self._screws_position = xys
        plt.scatter(xys[:, 0], xys[:, 1], s=1)

    def draw_strings(self, strings: List[List[int]]) -> None:
        assert self._screws_position is not None, "need to draw screws first."
        for line in strings:
            self._draw_screws_string_connection(line[0], line[1])

    def _draw_screws_string_connection(self, screw1: int, screw2: int) -> None:
        pos_1 = self._screws_position[screw1]
        pos_2 = self._screws_position[screw2]
        plt.plot([pos_1[0], pos_2[0]], [pos_1[1], pos_2[1]], alpha=0.17, c='black', linewidth=0.5)

    def show(self):
        plt.gcf().gca().set_aspect('equal', 'datalim')
        plt.show()


if __name__ == '__main__':
    ui = Infrastructure()
    ui.draw_screws()
    ui.draw_strings([[0, 4], [4, 20], [20, 100], [100, 10]])
    ui.show()
