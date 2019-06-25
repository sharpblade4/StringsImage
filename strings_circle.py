#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Union
from scipy import interpolate
from PIL import Image
import time

RADIUS = 50  # cm
SCREWS_AMOUNT = 470
STRING_LENGTH = 5.5 * 100 * 1000  # cm
STRING_LENGTH = 5.5 * 100  # TODO delete (testing)


class Gui:
    def __init__(self, screws_position: List[Tuple[float, float]]):
        self._screws_position = screws_position

    def draw_strings(self, strings: List[Tuple[int, int]]) -> None:
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
    def __init__(self, image_path: str):
        self._screws_position = None
        self._image = None
        self._distances = None
        self._samples = None
        self._prepare_image(image_path)
        self._init_screws()
        self._init_distances_and_samples()

    def _prepare_image(self, image_path: str) -> None:
        # load image as grayscale
        im = np.array(Image.open(image_path).convert('L'))
        # crop image to circle center
        center = np.floor(0.5 * np.array(im.shape)).astype(np.int32)
        im_radius = min(center)
        im_crop = im[center[0] - im_radius: center[0] + im_radius,
                  center[1] - im_radius: center[1] + im_radius]
        # scale to circle size
        im = np.array(Image.fromarray(im_crop).resize((RADIUS * 2 + 1, RADIUS * 2 + 1)))
        im = (100 / 255) * im  # TODO consider
        # plt.imshow(im)  # TODO delete (testing)
        self._image = im.copy()
        # TODO consider quantisizing, blurring, normalizing to gray

    def _init_screws(self) -> None:
        x0, y0 = np.array(self._image.shape[::-1]) * 0.5
        rads = (2 * np.pi / SCREWS_AMOUNT) * np.arange(SCREWS_AMOUNT)
        xys = np.array([[x0 + RADIUS * np.sin(angle), y0 + RADIUS * np.cos(angle)]
                        for angle in rads])
        self._screws_position = xys

    def _init_distances_and_samples(self) -> None:
        assert self._screws_position is not None, "screws not initialized"
        all_screws_pos = np.array(self._screws_position)
        all_screws_x_diff = np.subtract.outer(all_screws_pos[:, 0], all_screws_pos[:, 0])
        all_screws_y_diff = np.subtract.outer(all_screws_pos[:, 1], all_screws_pos[:, 1])
        self._distances = np.sqrt(np.sum(np.power(np.stack((all_screws_x_diff, all_screws_y_diff)), 2), axis=0))

        # TODO can I vectorise this? (preprocess optimization is less crucial).
        self._samples = {}
        for i1, p1 in enumerate(self._screws_position):
            for i2, p2 in enumerate(self._screws_position):
                if i1 == i2:
                    continue
                euclidean_distance = self._distances[i1, i2]
                sample_rate = np.arange(np.floor(euclidean_distance - 1)) / euclidean_distance
                xs = np.hstack([p1[0] + (p2[0] - p1[0]) * sample_rate, p2[0]])
                ys = np.hstack([p1[1] + (p2[1] - p1[1]) * sample_rate, p2[1]])
                self._samples[(i1, i2)] = (xs, ys)

    def get_distance(self, screw1: int, screw2: int) -> float:
        if not (0 <= screw1 < SCREWS_AMOUNT) or \
                not (0 <= screw2 < SCREWS_AMOUNT):
            raise IndexError
        return self._distances[screw1, screw2]

    def sample_line(self, screw1: int, screw2: int) -> Tuple[List[float], List[float]]:
        if not (0 <= screw1 < SCREWS_AMOUNT) or \
                not (0 <= screw2 < SCREWS_AMOUNT):
            raise IndexError
        return self._samples[(screw1, screw2)]

    def get_screws_positions(self) -> List[Tuple[float, float]]:  # each screw's (x,y)
        assert self._screws_position is not None, "screws not initialized"
        return self._screws_position.copy()

    def get_image(self):
        return self._image

    @staticmethod
    def just_try(randomize: bool, connected: bool) -> Union[List[List[int]], List[Tuple[int, int]]]:
        if randomize:
            amount = 500
            if connected:
                return Engine.steps_to_tuples(np.random.randint(SCREWS_AMOUNT, size=amount))
            else:
                return np.random.randint(SCREWS_AMOUNT, size=(amount, 2))
        else:
            jump = 70
            return np.vstack([np.arange(SCREWS_AMOUNT - jump),
                              np.arange(SCREWS_AMOUNT - jump) + jump]).T


    @staticmethod
    def steps_to_tuples(steps: List[int]) -> List[Tuple[int, int]]:
        return list(zip(steps[:-1], steps[1:]))


class Algo:  # TODO separate all classes to different files.
    def __init__(self, engine):
        self._leftover_string = STRING_LENGTH
        self._engine = engine
        self._curr_state = engine.get_image().copy()
        self._steps = None

    def _apply_string(self, from_screw: int, to_screw: int) -> None:
        amount = self._engine.get_distance(from_screw, to_screw)
        xs, ys = self._engine.sample_line(from_screw, to_screw)
        self._leftover_string -= amount
        if self._leftover_string < 0:
            print("raise Exception('Used too much string')")  # TODO decide
            self._leftover_string -= amount
            return
        ys = np.round(ys).astype(np.int)
        if 100 in ys:
            print(1)
        if 100 in xs:
            print(1)
        # ys[np.where(ys >= self._curr_state.shape[0])] = self._curr_state.shape[0] - 1 TODO del
        xs = np.round(xs).astype(np.int)
        # xs[np.where(xs >= self._curr_state.shape[1])] = self._curr_state.shape[1] - 1  TODO del
        self._curr_state[ys, xs] += 1
        # for point in zip(xs, ys):  # TODO optimize
        #     col1, col2 = int(np.floor(point[0])), int(np.ceil(point[0]))
        #     if col2 >= self._curr_state.shape[1]:
        #         col2 = col1
        #     if col1 == col2:
        #         col1_val = 1
        #         col2_val = 0
        #     else:
        #         col1_val = point[0] - col1
        #         col2_val = col2 - point[0]
        #     row1, row2 = int(np.floor(point[1])), int(np.ceil(point[1]))
        #     if row2 >= self._curr_state.shape[0]:
        #         row2 = row1
        #     if row1 == row2:
        #         row1_val = 1
        #         row2_val = 0
        #     else:
        #         row1_val = point[1] - row1
        #         row2_val = row2 - point[1]
        #     if col1 >= self._curr_state.shape[1] or row1 >= self._curr_state.shape[0]:
        #         print('bad...', col1, col2, row1, row2, from_screw, to_screw)
        #         continue  # TODO FIXME
        #     self._curr_state[row1, col1] += 0.5 * (col1_val + row1_val)
        #     self._curr_state[row1, col2] += 0.5 * (col2_val + row1_val)
        #     self._curr_state[row2, col2] += 0.5 * (col2_val + row2_val)
        #     self._curr_state[row2, col1] += 0.5 * (col1_val + row2_val)
        print('\tleftover:', self._leftover_string)  # TODO delete

    def get_next(self, current_screw: int, grid) -> Tuple[int, float]:
        if not (0 <= current_screw < SCREWS_AMOUNT):
            raise IndexError
        best_candidate = current_screw
        while best_candidate == current_screw:
            best_candidate = (current_screw + np.random.randint(SCREWS_AMOUNT)) % SCREWS_AMOUNT
        best_score = self._score_line(current_screw, best_candidate, grid)
        for screw_i in range(SCREWS_AMOUNT):
            if screw_i in (current_screw, best_candidate):
                continue
            score = self._score_line(current_screw, screw_i, grid)
            if score > best_score:
                best_candidate = screw_i
                best_score = score
        return best_candidate, best_score

    def _score_path(self, degree: int, current_screw: int) -> Tuple[int, List[float],
                                                                    List[float]]:
        im_height, im_width = self._curr_state.shape
        grid = interpolate.interp2d(np.arange(im_width), np.arange(im_height),
                                    self._curr_state)
        for i in range(degree):
            best_candidate = self.get_next(current_screw)

    def _score_line(self, screw1: int, screw2: int, grid) -> float:
        # TODO optimize by RectBivariateSpline
        xs, ys = self._engine.sample_line(screw1, screw2)
        if len(xs) > 1 or len(ys) > 1:
            intensities = grid(xs, ys).diagonal()
        else:
            intensities = grid(xs, ys)
        score = (-1 * np.sum(intensities)) / self._engine.get_distance(screw1, screw2)
        return score

    def execute(self) -> List[int]:
        im_height, im_width = self._curr_state.shape
        grid = interpolate.interp2d(np.arange(im_width), np.arange(im_height),
                                    self._curr_state)
        current_screw = np.random.randint(SCREWS_AMOUNT)
        steps = [current_screw]
        while self._leftover_string > 0:
            next_screw, _ = self.get_next(current_screw, grid)
            self._apply_string(current_screw, next_screw)
            steps.append(next_screw)
            current_screw = next_screw
            grid = interpolate.interp2d(np.arange(im_width), np.arange(im_height),
                                        self._curr_state)
        return steps


def main(image_path):
    print("Running with string length", STRING_LENGTH / (100 * 1000),
          "km, to create a circle with radius", RADIUS, 'cm, with', SCREWS_AMOUNT,
          'screws.')
    infrastructure_engine = Engine(image_path)
    ui = Gui(infrastructure_engine.get_screws_positions())

    begin_time = time.time()
    algo = Algo(infrastructure_engine)
    res_steps = algo.execute()
    print('Time: ', time.time() - begin_time)
    print(res_steps)

    # trial = infrastructure_engine.just_try(randomize=True, connected=True)
    ui.draw_screws()
    ui.draw_strings(infrastructure_engine.steps_to_tuples(res_steps))
    ui.show()


if __name__ == '__main__':
    tux_path = '/home/ru/Pictures/tux-100677393-large.jpg'
    half_black = '/home/ru/Pictures/halfblack.jpg'
    # main(tux_path)
    main(half_black)
