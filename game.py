DRAW = True

if DRAW:
    import pygame
    from pygame.locals import *

from numpy import *
import numpy.random as nrand
import random
import os

#define
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

        if self.draw:
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            pygame.init()
            self.surface = pygame.Surface((width * tile_size, height * tile_size))
            self.clock = pygame.time.Clock()
            size = (tile_size, tile_size)
            self.images = {}
            self.images[COVERED] = pygame.transform.scale(self.loadImg("covered"), size)
            for i in range(INCORRECT + 1):
                self.images[i] = pygame.transform.scale(self.loadImg(str(i)), size)

            self.screen = pygame.display.set_mode((width * tile_size, height * tile_size))

        self.reset()

    def loadImg(self, name):
        return pygame.image.load(os.path.join("images", name + ".png"))

    def drawTile(self, pos, val):
        if self.draw:
            y, x = pos
            p = (y * self.tileSize, x * self.tileSize)
            self.surface.blit(self.images[val], p)
            self.screen.blit(self.surface, (0, 0))

    def reset(self):
        w = self.width
        h = self.height
        self.display_board = [COVERED for i in range(self.wh)]

        self.board = [[0 for i in range(w)] for j in range(h)]

        mines = 0
        while (mines != self.mines):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            if (x != 0 or y != 0) and (self.board[y][x] != BOMB):
                self.board[y][x] = BOMB
                mines += 1

        for i in range(h):
            for j in range(w):
                self.drawTile((i, j), -1)
                if (self.board[i][j] == BOMB):
                    for i_inc in range(-1, 2):
                        for j_inc in range(-1, 2):
                            y_new = i + i_inc
                            x_new = j + j_inc
                            if y_new >= 0 and x_new >= 0:
                                try:
                                    if self.board[y_new][x_new] != BOMB:
                                        self.board[y_new][x_new] += 1
                                except:
                                    pass

        self.correct_flags = 0
        self.incorrect_flags = 0
        self.opened = 0
        self.finished = False
        self.result = None

        caption = "W: " + str(self.wins) + " - L: " + str(self.losses)
        if self.draw:
            pygame.display.set_caption(caption)
        print caption

    def open(self, pos, recursion = False):
        y, x = pos
        if self.display_board[y * self.width + x] == -1:
            if self.board[y][x] == BOMB and not recursion:
                self.display_board[y * self.width + x] = BOOM
                self.drawTile(pos, self.display_board[y * self.width + x])
                for h in range(self.height):
                    for w in range(self.width):
                        if self.display_board[h * self.width + w] == COVERED:
                            self.display_board[h * self.width + w] = self.board[h][w]
                        elif self.display_board[h * self.width + w] == FLAG and self.board[h][w] == BOMB:
                            self.display_board[h * self.width + w] = CORRECT
                        else:
                            self.display_board[h * self.width + w] = INCORRECT

                        self.drawTile((h, w), self.display_board[h * self.width + w])

                self.finished = True
                self.losses += 1
                self.result = False

            elif self.board[y][x] == 0:
                self.display_board[y * self.width + x] = 0
                self.opened += 1
                self.drawTile(pos, self.display_board[y * self.width + x])

                for i_inc in range(-1, 2):
                    for j_inc in range(-1, 2):
                        y_new = y + i_inc
                        x_new = x + j_inc
                        if (i_inc != 0 or j_inc != 0) and (y_new >= 0 and x_new >= 0):
                            try:
                                self.open((y_new, x_new), True)
                            except:
                                pass

            elif self.board[y][x] != BOMB:
                self.display_board[y * self.width + x] = self.board[y][x]
                self.drawTile(pos, self.display_board[y * self.width + x])

            else:
                pass # we encountered bomb in the reccursion, don't open the tile0

            self.checkWin()

    def mark(self, pos):
        y, x = pos
        if self.display_board[y * self.width + x] == COVERED:
            self.display_board[y * self.width + x] = FLAG
            if self.board[y][x] == BOMB:
                self.correct_flags += 1
            else:
                self.incorrect_flags += 1
        elif self.display_board[y * self.width + x] == FLAG:
            self.display_board[y * self.width + x] = COVERED
            if self.board[y][x] == BOMB:
                self.correct_flags -= 1
            else:
                self.incorrect_flags -= 1

        self.drawTile(pos, self.display_board[y * self.width + x])

    def checkWin(self):
        if not self.finished:
            if (self.opened == self.wh - self.mines):
                self.finished = True
                self.result = True
                self.wins += 1
                return True
        return self.result

    def isDone(self):
        return not self.finished