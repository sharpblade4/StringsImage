#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Union
from PIL import Image
import time
import os

RADIUS = 150  # aim to 20 cm (/7.5)
SCREWS_AMOUNT = 210  # better resolution in 470
STRING_LENGTH = 7.5 * RADIUS * 615
THREAD_WEIGHT = 13
DISTANCE_WEIGHT = 20.7


class Gui:
    def __init__(self, screws_position: List[Tuple[float, float]]):
        self._screws_position = screws_position

    def draw_strings(self, strings: List[Tuple[int, int]]) -> None:
        for line in strings:
            self._draw_screws_string_connection(line[0], line[1])

    def draw_screws(self, labels=False) -> None:
        plt.scatter(self._screws_position[:, 0], self._screws_position[:, 1], s=1)
        if not labels:
            return
        for i, sc in enumerate(self._screws_position[::10]):
            plt.text(sc[0], sc[1], str(i * 10))

    def _draw_screws_string_connection(self, screw1: int, screw2: int) -> None:
        pos_1 = self._screws_position[screw1]
        pos_2 = self._screws_position[screw2]
        plt.plot([pos_1[0], pos_2[0]], [pos_1[1], pos_2[1]],
                 alpha=0.99, c='black', linewidth=0.233)

    def show(self) -> None:
        plt.axis('off')
        plt.gcf().tight_layout()
        plt.gca().invert_yaxis()
        plt.gcf().gca().set_aspect('equal', 'datalim')
        plt.show()


class Engine:
    def __init__(self, image_path: str, preprocess: bool = True):
        self._screws_position = None
        self._image = None
        self._distances = None
        self._samples = None
        self._prepare_image(image_path)
        self._init_screws()
        if preprocess:
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
        im = (100 / 255) * im
        self._image = im.copy()
        # TODO consider quantisizing, blurring, normalizing to gray

    def _init_screws(self) -> None:
        x0, y0 = np.array(self._image.shape[::-1]) * 0.5
        rads = (2 * np.pi / SCREWS_AMOUNT) * np.arange(SCREWS_AMOUNT)
        xys = np.array([[x0 + RADIUS * np.sin(angle), y0 + RADIUS * np.cos(angle)]
                        for angle in rads])
        self._screws_position = xys

    def _init_distances_and_samples(self) -> None:
        preprocessed_path = f'preprocessed_{RADIUS}_{SCREWS_AMOUNT}.npy'
        if os.path.exists(preprocessed_path):
            loaded_preprocessed = np.load(preprocessed_path, allow_pickle=True)
            self._distances, self._samples = loaded_preprocessed
            print('Working with preprocessed data.')
        else:
            assert self._screws_position is not None, 'screws not initialized'
            all_screws_pos = np.array(self._screws_position)
            all_screws_x_diff = np.subtract.outer(all_screws_pos[:, 0], all_screws_pos[:, 0])
            all_screws_y_diff = np.subtract.outer(all_screws_pos[:, 1], all_screws_pos[:, 1])
            self._distances = np.sqrt(np.sum(np.power(np.stack((all_screws_x_diff, all_screws_y_diff)), 2), axis=0))

            # TODO vectorise? (preprocess optimization is less crucial).
            self._samples = {}
            for i1, p1 in enumerate(self._screws_position):
                for i2, p2 in enumerate(self._screws_position[i1 + 1:], i1 + 1):
                    euclidean_distance = self._distances[i1, i2]
                    sample_rate = np.arange(np.floor(euclidean_distance - 1)) / euclidean_distance
                    xs = np.hstack([p1[0] + (p2[0] - p1[0]) * sample_rate, p2[0]])
                    ys = np.hstack([p1[1] + (p2[1] - p1[1]) * sample_rate, p2[1]])
                    self._samples[(i1, i2)] = (xs, ys)
            np.save(preprocessed_path[:-4], (self._distances, self._samples))
            print('saved preprocessed data.')

    def get_distance(self, screw1: int, screw2: int) -> float:
        if not (0 <= screw1 < SCREWS_AMOUNT) or \
                not (0 <= screw2 < SCREWS_AMOUNT):
            raise IndexError
        # s1, s2 = min(screw1, screw2), max(screw1, screw2)  # no need, profit is low.
        return self._distances[screw1, screw2]

    def sample_line(self, screw1: int, screw2: int) -> Tuple[List[float], List[float]]:
        if not (0 <= screw1 < SCREWS_AMOUNT) or \
                not (0 <= screw2 < SCREWS_AMOUNT):
            raise IndexError
        s1, s2 = min(screw1, screw2), max(screw1, screw2)
        return self._samples[(s1, s2)]

    def get_screws_positions(self) -> List[Tuple[float, float]]:  # each screw's (x,y)
        assert self._screws_position is not None, 'screws not initialized'
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


# TODO separate all classes to different files.

class Algo:
    def __init__(self, engine):
        self._leftover_string = STRING_LENGTH
        self._engine = engine
        self._curr_state = engine.get_image().copy()
        self._steps = None

    def _update_state(self, from_screw: int, to_screw: int, state):
        xs, ys = self._engine.sample_line(from_screw, to_screw)
        ys = np.round(ys).astype(np.int)
        xs = np.round(xs).astype(np.int)
        ys = ys.clip(0, state.shape[0] - 1)
        xs = xs.clip(0, state.shape[1] - 1)
        edited_state = state.copy()
        edited_state[ys, xs] += THREAD_WEIGHT
        return edited_state

    def _get_next_naive(self, current_screw: int, state) -> Tuple[int, float]:
        if not (0 <= current_screw < SCREWS_AMOUNT):
            raise IndexError
        best_candidate = current_screw
        while best_candidate == current_screw:
            best_candidate = (current_screw + np.random.randint(SCREWS_AMOUNT)) % SCREWS_AMOUNT
        best_score = self._score_line_naive(current_screw, best_candidate, state)
        for screw_i in range(SCREWS_AMOUNT):
            if screw_i in (current_screw, best_candidate):
                continue
            score = self._score_line_naive(current_screw, screw_i, state)
            if score > best_score:
                best_candidate = screw_i
                best_score = score
        return best_candidate, best_score

    def _score_path_aux(self, degree: int, current_screw: int, next_screw: int,
                        state) -> Tuple[float, float, np.ndarray, List[int]]:
        state = self._update_state(current_screw, next_screw, state)
        cur2next_score = self._score_line_naive(current_screw, next_screw, state)
        cur2next_amount = self._engine.get_distance(current_screw, next_screw)
        score, amount, c_state, c_screws = self._get_path(degree - 1, next_screw, state)
        return cur2next_score + score, cur2next_amount + amount, c_state, [next_screw] + c_screws

    def _get_path(self, degree: int, current_screw: int, state) -> Tuple[float, float, np.ndarray, List[int]]:
        # recursively get the best next and returns (accumulative score, accumulative amount, state, final screw)
        assert degree > 0, f'Algo::_get_path called with invalid degree: {degree}'
        if degree == 1:
            following_screw, following_score = self._get_next_naive(current_screw, state)
            assert following_screw != current_screw
            following_state = self._update_state(current_screw, following_screw, state)
            following_amount = self._engine.get_distance(current_screw, following_screw)
            return following_score, following_amount, following_state, [following_screw]
        else:
            # typedef:   b_ prefix for best_something;  c_ prefix for candidate_something.
            best_candidate = current_screw
            while best_candidate == current_screw:
                best_candidate = (current_screw + np.random.randint(SCREWS_AMOUNT)) % SCREWS_AMOUNT
            b_score, b_amount, b_state, b_screws = self._score_path_aux(degree, current_screw, best_candidate, state)
            for screw_i in range(SCREWS_AMOUNT):
                if screw_i in (current_screw, best_candidate):
                    continue
                c_score, c_amount, c_state, c_screws = self._score_path_aux(degree, current_screw, screw_i, state)
                if c_score > b_score:
                    best_candidate = screw_i
                    b_score, b_amount, b_state, b_screws = c_score, c_amount, c_state, c_screws
            assert len(b_screws) == degree
            return b_score, b_amount, b_state, b_screws

    def _score_line_naive(self, screw1: int, screw2: int, state) -> float:
        xs, ys = self._engine.sample_line(screw1, screw2)
        ys = np.round(ys).astype(np.int)
        xs = np.round(xs).astype(np.int)
        ys = ys.clip(0, state.shape[0] - 1)
        xs = xs.clip(0, state.shape[1] - 1)
        return (-1 * np.sum(state[ys, xs])) / (DISTANCE_WEIGHT*self._engine.get_distance(screw1, screw2))

    def execute(self, degree: int) -> List[int]:
        current_screw = np.random.randint(SCREWS_AMOUNT)
        next_screws = [current_screw]
        steps = []
        modulus = 7 - (degree ** 2)
        while self._leftover_string > 0:
            score, amount, next_state, next_screws = self._get_path(degree, current_screw=current_screw,
                                                                    state=self._curr_state)
            steps.extend([current_screw] + next_screws[:-1])
            current_screw = next_screws[-1]
            self._leftover_string -= amount
            self._curr_state = next_state
            if len(steps) % modulus == 0:
                print(f'leftover: {self._leftover_string}')
        steps.append(next_screws[-1])
        return steps


def main_load(image_path: str, steps_path: str):
    infrastructure_engine = Engine(image_path, preprocess=False)
    steps = np.load(steps_path)
    steps = steps[1000:]
    steps = np.mod(steps + 157, 210)  # steps = np.roll(steps, 157)
    tup_steps = infrastructure_engine.steps_to_tuples(steps)
    print(len(steps))
    ui = Gui(infrastructure_engine.get_screws_positions())
    ui.draw_screws()
    ui.draw_strings(tup_steps)
    ui.show()
    stxt = 209 - steps
    with open('out_steps.txt', 'w') as f:
        f.write(', '.join([str(x) for x in stxt]))
    print(stxt)


def main(image_path: str, calculate_only: bool = False):
    print(f'Running with string length {STRING_LENGTH / (7.5 * 100 * 1000)} km, to create a circle with radius '
          f'{RADIUS / 6} cm, with {SCREWS_AMOUNT} screws.')
    infrastructure_engine = Engine(image_path)
    ui = Gui(infrastructure_engine.get_screws_positions())

    algo = Algo(infrastructure_engine)
    begin_time = time.time()
    res_steps = algo.execute(1)
    endtime = time.time()
    np.save(f'last_run_{STRING_LENGTH / (7.5 * 100 * 1000)}_{RADIUS / 7.5}_{SCREWS_AMOUNT}', res_steps)
    print('Algo time: ', endtime - begin_time)
    print('Steps amount: ', len(res_steps))

    # trial = infrastructure_engine.just_try(randomize=True, connected=True)
    if not calculate_only:
        ui.draw_screws()
        # plt.imshow(infrastructure_engine.get_image())
        ui.draw_strings(infrastructure_engine.steps_to_tuples(res_steps))
        ui.show()


if __name__ == '__main__':
    my_photo = '/home/ru/Pictures/myphotox2.jpg'
    main(my_photo)
    # npy_steps_path = '/home/ru/develop/StringsImage/list146.npy'
    # final_npy_steps_path = '/home/ru/develop/StringsImage/mypicture34.pnglast_run_3.0_25.0_210.npy'
    # main_load(my_photo, final_npy_steps_path)
