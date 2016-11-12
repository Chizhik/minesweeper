# DRAW = True

# if DRAW:
#     import pygame
#     from pygame.locals import *

import numpy as np
import numpy.random as nrand
import random
import os

#define
TEST = True
COVERED = -1
FLAG = 9
BOMB = 10
CORRECT = 11
INCORRECT = 12
BOOM = 13
#end define

class Game(object):
    def __init__(self, width, height, mines, draw = False, tile_size = 32):
        self.width = width
        self.height = height
        self.mines = mines
        self.wins = 0
        self.losses = 0
        self.draw = draw
        self.tileSize = tile_size
        self.wh = self.height * self.width

        self.reset()


    def reset(self):
        w = self.width
        h = self.height
        self.display_board = np.zeros((h, w))
        self.display_board += COVERED

        self.board = np.zeros((h, w))

        mines = 0
        while (mines != self.mines):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            if (x != 0 or y != 0) and (self.board[y,x] != BOMB):
                self.board[y,x] = BOMB
                mines += 1

        for i in range(h):
            for j in range(w):
                if (self.board[i,j] == BOMB):
                    for i_inc in range(-1, 2):
                        for j_inc in range(-1, 2):
                            y_new = i + i_inc
                            x_new = j + j_inc
                            if y_new >= 0 and x_new >= 0:
                                try:
                                    if self.board[y_new,x_new] != BOMB:
                                        self.board[y_new,x_new] += 1
                                except:
                                    pass

        self.correct_flags = 0
        self.incorrect_flags = 0
        self.opened = 0
        self.finished = False
        self.result = None

        caption = "W: " + str(self.wins) + " - L: " + str(self.losses)

        print caption

        return np.matrix(self.display_board.flatten()) + 1

    def open_recursive(self, y, x):
        if self.display_board[y,x] == -1:
            if self.board[y,x] == 0:
                self.display_board[y,x] = 0
                self.opened += 1

                for i_inc in range(-1, 2):
                    for j_inc in range(-1, 2):
                        y_new = y + i_inc
                        x_new = x + j_inc
                        if (i_inc != 0 or j_inc != 0) and (y_new >= 0 and x_new >= 0):
                            try:
                                self.open_recursive(y_new, x_new)
                            except:
                                pass

            elif self.board[y,x] != BOMB:
                self.display_board[y,x] = self.board[y,x]
                self.opened += 1

            else:
                pass # we encountered bomb in the reccursion, don't open the tile0

    def open(self, pos):
        y, x = pos
        reward = 1.0
        done = False
        
        if self.display_board[y,x] != -1:
            reward = -1.0
            done = True
        
        elif self.board[y,x] == BOMB:
            self.display_board[y,x] = BOOM
            for h in range(self.height):
                for w in range(self.width):
                    if self.display_board[h,w] == COVERED:
                        self.display_board[h,w] = self.board[h][w]
                    elif self.display_board[h,w] == FLAG and self.board[h][w] == BOMB:
                        self.display_board[h,w] = CORRECT
                    elif self.display_board[h,w] == FLAG and self.board[h][w] != BOMB:
                        self.display_board[h,w] = INCORRECT

            reward = -1.0
            self.finished = True
            self.losses += 1
            self.result = False
            done = True
            
        else:
            self.open_recursive(y, x)
            

        if self.checkWin():
            reward = 10.0
            done = True
        if TEST:
            print pos
            print self.display_board
        return np.matrix(self.display_board.flatten()) + 1, reward, done 

    def mark(self, y, x):
        if self.display_board[y,x] == COVERED:
            self.display_board[y,x] = FLAG
            if self.board[y,x] == BOMB:
                self.correct_flags += 1
            else:
                self.incorrect_flags += 1
        elif self.display_board[y,x] == FLAG:
            self.display_board[y,x] = COVERED
            if self.board[y,x] == BOMB:
                self.correct_flags -= 1
            else:
                self.incorrect_flags -= 1
        if TEST:
            print self.display_board


    def checkWin(self):
        if not self.finished:
            if (self.opened == self.wh - self.mines):
                self.finished = True
                self.result = True
                self.wins += 1
                return True
        return self.result

    def isDone(self):
        return self.finished