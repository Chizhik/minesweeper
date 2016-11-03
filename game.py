DRAW = True

if DRAW:
    import pygame
    from pygame.locals import *

from numpy import *
import numpy.random as nrand
import random

class Game(object):
    def __init__(self, width, height, mines, draw = False, tile_size = 32):
        self.width = width
        self.height = height
        self.mines = mines
        self.wins = 0
        self.losses = 0
        self.draw = draw

        if self.draw:
            pygame.init()
            self.surface = pygame.Surface((width * tile_size, height * tile_size))
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        w = self.width
        h = self.height
        self.display_board = [-1 for i in range(w * h)]

        self.board = [[0 for i in range(w)] for j in range(h)]

        mines = 0
        while (mines != self.mines):
            x = random.randint(0, w)
            y = random.randint(0, h)
            if (x != 0 or y != 0) and (self.board[y][x] != 10):
                self.board[y][x] = 10
                mines += 1

        for i in range(h):
            for j in range(w):
                if (self.board[i][j] != 10):
                    for i_inc in range(-1, 2):
                        for j_inc in range(-1, 2):
                            try:
                                if self.board[i + i_inc][j + j_inc] != 10:
                                    self.board[i + i_inc][j + j_inc] += 1
                            except:
                                pass

        self.correct_flags = 0
        self.incorrect_flags = 0