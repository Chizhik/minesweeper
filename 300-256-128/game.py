import numpy as np
import numpy.random as nrand
import random
import os

# define
TEST = False
COVERED = -1
FLAG = 9
BOMB = 10
CORRECT = 11
INCORRECT = 12
BOOM = 13
# end define


class Game(object):
    def __init__(self, width, height, mines, draw=True, tile_size=32):
        self.width = width
        self.height = height
        self.mines = mines
        self.wins = 0
        self.losses = 0
        self.draw = draw
        self.tileSize = tile_size
        self.wh = self.height * self.width

        # gamepy setup
        if self.draw:
            global pygame
            import pygame
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            pygame.init()
            self.surface = pygame.Surface((width*tile_size, height*tile_size))
            self.screen = pygame.display.set_mode((width*tile_size, height*tile_size))
            self.clock = pygame.time.Clock()
            self.pics = {}
            size = (self.tileSize, self.tileSize)
            self.pics[-1] = pygame.transform.scale(self.loadImg("covered"), size)
            for i in range(14):
                self.pics[i] = pygame.transform.scale(self.loadImg(str(i)), size)

            self.wt = True

        self.reset()

    def loadImg(self, name):
        return pygame.image.load(os.path.join("icons", name + ".png"))

    def drawTile(self, pos, val):
        if self.draw:
            y, x = pos
            p = (x * self.tileSize, y * self.tileSize)
            self.surface.blit(self.pics[val], p)

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
            if (x != 0 or y != 0) and (self.board[y, x] != BOMB):
                self.board[y, x] = BOMB
                mines += 1
                pass

        for i in range(h):
            for j in range(w):
                if self.board[i,j] == BOMB:
                    for y in range(max(0, i - 1), min(i+2, w)):
                        for x in range(max(0, j - 1), min(j+2, h)):
                            if self.board[y, x] != BOMB:
                                self.board[y, x] += 1

        self.correct_flags = 0
        self.incorrect_flags = 0
        self.opened = 0
        self.finished = False
        self.result = None

        caption = "W: " + str(self.wins) + " - L: " + str(self.losses)
        if self.draw:
            pygame.display.set_caption(caption)
            for i in range(h):
                for j in range(w):
                    p = (i * self.tileSize, j * self.tileSize)
                    self.surface.blit(self.pics[-1], p)
            self.screen.blit(self.surface, (0,0))
            pygame.display.flip()

        print caption

        return np.matrix(self.display_board.flatten()) + 1

    def open_recursive(self, y, x):
        if self.display_board[y, x] == COVERED:
            if self.board[y, x] == 0:
                self.display_board[y, x] = 0
                self.opened += 1

                for y_new in range(max(0, y - 1), min(y + 2, self.width)):
                    for x_new in range(max(0, x - 1), min(x + 2, self.height)):
                        self.open_recursive(y_new, x_new)

            elif self.board[y, x] != BOMB:
                self.display_board[y, x] = self.board[y, x]
                self.opened += 1

            else:
                pass  # we encountered bomb in the reccursion, don't open the tile0


        if self.draw:
            self.drawTile((y,x), self.display_board[y, x])


    def open(self, pos):
        y, x = pos
        reward = 1.0
        done = False

        if self.draw:
            self.pause()

        if self.display_board[y, x] != COVERED:
            reward = -1.0
            done = True

        elif self.board[y, x] == BOMB:
            self.display_board[y, x] = BOOM
            for h in range(self.height):
                for w in range(self.width):
                    if self.display_board[h, w] == COVERED:
                        self.display_board[h, w] = self.board[h][w]
                    elif self.display_board[h, w] == FLAG and self.board[h][w] == BOMB:
                        self.display_board[h, w] = CORRECT
                    elif self.display_board[h, w] == FLAG and self.board[h][w] != BOMB:
                        self.display_board[h, w] = INCORRECT

                    self.drawTile((h,w), self.display_board[h, w])

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
            print self.display_board

        if self.draw:
            self.drawTile(pos, self.display_board[y, x])
            self.screen.blit(self.surface, (0, 0))

            tran = pygame.Surface((self.tileSize - 1, self.tileSize - 1))
            tran.fill((255, 255, 255))
            tran.set_alpha(170)
            self.screen.blit(tran, (x * self.tileSize, y * self.tileSize))

            pygame.display.flip()

            self.pause()


        return np.matrix(self.display_board) + 1, reward, done

    def mark(self, pos):
        y, x = pos
        if self.display_board[y, x] == COVERED:
            self.display_board[y, x] = FLAG
            if self.board[y, x] == BOMB:
                self.correct_flags += 1
            else:
                self.incorrect_flags += 1

        if TEST:
            print self.display_board

        if self.draw:
            self.drawTile((y,x), self.display_board[y, x])
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()


    def unmark(self, pos):
        y, x = pos
        if self.display_board[y, x] == FLAG:
            self.display_board[y, x] = COVERED
            if self.board[y, x] == BOMB:
                self.correct_flags -= 1
            else:
                self.incorrect_flags -= 1

        if self.draw:
            self.drawTile((y,x), self.display_board[y, x])
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

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

    def pause(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.wt = not self.wt
                elif event.key == pygame.K_ESCAPE:
                    exit()

        if self.wt:
            event = pygame.event.wait()
            while (1):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.wt = not self.wt
                        break
                    elif event.key == pygame.K_RETURN:
                        break
                    elif event.key == pygame.K_ESCAPE:
                        exit()
                event = pygame.event.wait()

    def fillProbabilities(self, pmines, pboard):
        if self.draw:
            tran = pygame.Surface((self.tileSize - 1, self.tileSize - 1))
            for i in range(len(pmines)):
                h, w = pmines[i]
                g = 1 - pboard[h, w]
                g *= 255
                tran.fill((160, 255 - g, g))
                tran.set_alpha(120)
                self.screen.blit(tran, (w * self.tileSize, h * self.tileSize))

            pygame.display.flip()