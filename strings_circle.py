#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

RADIUS = 50  # cm
SCREWS_AMOUNT = 470
STRING_LENGTH = 5.5 * 100 * 1000  # cm


class Infrastructure:  # TODO refactor to Gui
    def __init__(self, screws_position):
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
        # TODO full screen + tight layout + hide axises
        plt.gcf().gca().set_aspect('equal', 'datalim')
        plt.show()

        
class Engine:
    def __init__(self):
        self._leftover_string = STRING_LENGTH
        self._screws_position = None
        self.init_screws()
        # TODO Engine should handle all image processing (blurring, quantisizing)
    
    def init_screws(self) -> None:  # TODO maybe get from outside
        x0, y0 = (0, 0)
        rads = (2 * np.pi / SCREWS_AMOUNT) * np.arange(SCREWS_AMOUNT)
        xys = np.array([[x0 + RADIUS * np.sin(angle), y0 + RADIUS * np.cos(angle)] 
                              for angle in rads])
        self._screws_position = xys
    
    def get_screws_positions(self): # TODO check: -> List[List[Tuple(float, float)]
        assert self._screws_position is not None, "screws not initialized"
        return self._screws_position.copy()
    
    def just_try(self, randomize) -> List[List[int]]:
        if randomize:
            amount = 500
            return np.random.randint(SCREWS_AMOUNT, size=(amount, 2))
        else:
            jump = 70
            return np.vstack([np.arange(SCREWS_AMOUNT - jump), 
                                      np.arange(SCREWS_AMOUNT - jump) + jump]).T

    def calculate_string_usage(self, screw1 : int, screw2 : int) -> float:
        if  not (0 <= screw1 < SCREWS_AMOUNT) or  \
            not (0 <= screw2 < SCREWS_AMOUNT):
            raise KeyError  # TODO check if there is a more appropriate exception
        p1 = np.array(self._screws_position[screw1])
        p2 = np.array(self._screws_position[screw2])
        euclidean_distance = np.sqrt(np.sum(np.power(p2-p1, 2)))
        return euclidean_distance
        
    def use_string(self, amount : float) -> None:
        if self._leftover_string - amount < 0:
            raise Exception("Used too much string")
        self._leftover_string -= amount
        
                                      
class Algo:  # TODO separate all classes to different files.
    def __init__(self, image, screws_position):
        self._screws_position = screws_position
        #TODO use a grid (scipy interpolate2D) on image. 
        self._orig_image = image
        self._curr_state = self._orig_image.copy()
    
    def get_next(self, current_screw : int): # TODO check: state : Dict, -> int:
        if not (0 <= current_screw < SCREWS_AMOUNT):
            raise KeyError # TODO check if there is a more appropriate exception
        # TODO implement
    
    def score_line(self, screw1 : int, screw2: int) -> float:
        pass
        # TODO implement (use grid)
        
        
        
                                      
if __name__ == '__main__':
    engine = Engine()
    ui = Infrastructure(engine.get_screws_positions())
    
    trial = engine.just_try(randomize=True)
    
    ui.draw_screws()
    ui.draw_strings(trial)
    ui.show()
