from game import *
from ExperienceBuf import *
from NN import *
import numpy as np
import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, load = False, disp = False):
        self.bsize = (10, 10)
        self.game = Game(10, 10, 10, disp)
        self.nn = NN(self.bsize)
        if load:
            self.nn.load()
        self.buf = ExperienceBuf()
        self.num_episodes = 500
        self.max_game_moves = 64
        self.observe = 10
        self.e = 0.0
        self.y = 0.99
        self.mini_batch = 64

    def toHot(self, inputState):
        temp = []
        s = np.array([])
        for i in range(25):
            sample = np.zeros(11)
            if (inputState[i] >= 0):
                sample[int(inputState[i])] = 1.0
            temp.append(sample)
        return np.asmatrix(np.append(s, temp))

    def is_uncovered(self, y, x):
        return 0 <= self.game.display_board[y, x] <= 8

    def is_in_range(self, y, x):
        return 0 <= y < self.bsize[0] and 0 <= x < self.bsize[1]


    def learn(self):
        count = 0

        for i in range(self.num_episodes):
            self.game.reset()
            if (self.game.wins + self.game.losses != i):
                break;
            _, _, d = self.game.open((0, 0))
            while (not d):
                self.place_flags()
                #pmines, true_vals = self.choose_possible_mines()
                pmines, true_vals = self.border_tiles_with_true_values()
                #print pmines
                #print true_vals

                if (len(pmines) == 0):
                    uncovered_list = self.all_uncovered()
                    #print uncovered_list
                    a = random.sample(uncovered_list, 1)[0]
                else:
                    states = self.make_states(pmines)
                    states.shape = (len(pmines), 1, 25)

                    pboard = np.zeros(self.bsize)

                    for i in range(len(pmines)):
                        tmp_state = states[i].flatten()
                        tmp_state = self.toHot(tmp_state)
                        self.buf.memorize(tmp_state, true_vals[i])
                        h, w = pmines[i]
                        pboard[h, w] = self.nn.predict(tmp_state)

                    # print pmines
                    # print states
                    # print pboard

                    act = np.argmax(pboard.flatten())
                    a = (act // self.bsize[1], act % self.bsize[1])



                # print a
                _, _, d = self.game.open(a)

                count += 1



            if count > self.observe:
                x, y, bsize = self.buf.get_batch(self.mini_batch)
                x = np.asarray(x)
                x.shape = (bsize, 275)
                y = np.asarray(y)
                y.shape = (bsize, 1)
                self.nn.train(x, y)

            if count % 50 == 0:
                self.nn.save()

        self.graph_stats()

    def graph_stats(self):
        plt.plot(self.nn.hist.losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('iter')
        plt.show()

        plt.plot(self.nn.hist.acc)
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('iter')
        plt.show()

    def play_solver(self):
        for i in range(self.num_episodes):
            self.game.reset()
            if (self.game.wins + self.game.losses != i):
                break;
            _, _, d = self.game.open((0, 0))
            while (not d):
                self.place_flags()
                pmines, true_vals = self.border_tiles_with_true_values()
                #print pmines
                #print true_vals
                if (len(pmines) == 0):
                    uncovered_list = self.all_uncovered()
                    #print uncovered_list
                    a = random.sample(uncovered_list, 1)[0]
                else:
                    a = pmines[np.argmax(true_vals)]
                # print a
                _, _, d = self.game.open(a)



    def make_target(self):
        target = np.zeros(self.bsize)
        s = self.game.display_board
        for h in range(self.bsize[0]):
            for w in range(self.bsize[1]):
                if s[h, w] == COVERED:
                    flag = False
                    for i in range(-1, 2):
                        h_new = h + i
                        if (h_new >= 0):
                            for j in range(-1, 2):
                                w_new = w + j
                                if (w_new >= 0):
                                    try:
                                        if (s[h_new, w_new] > 0):
                                            flag = True
                                    except:
                                        pass
                                if flag: break
                        if flag: break

                    if flag:
                        if self.game.board[h, w] == BOMB + 1: target[h, w] = 0
                        else: target[h, w] = 1
                    else: target[h, w] = 0.5
        return self.softmax(target)

    def choose_possible_mines(self):
        pmines = []
        true_vals = []
        s = self.game.display_board
        for h in range(self.bsize[0]):
            for w in range(self.bsize[1]):
                if s[h, w] == COVERED:
                    is_border = False
                    for i in range(-1, 2):
                        h_new = h + i
                        if (h_new >= 0 and h_new < 8):
                            for j in range(-1, 2):
                                w_new = w + j
                                if (w_new >= 0 and w_new < 8):
                                    if (s[h_new, w_new] > 0 and s[h_new, w_new] <= 8) or s[h_new, w_new] == FLAG:
                                        is_border = True
                                if is_border: break
                        if is_border: break

                    if is_border:
                        pmines.append((h, w))
                        if self.game.board[h, w] == BOMB: true_vals.append(0)
                        else:true_vals.append(1)
        return pmines, true_vals

    def border_tiles_with_true_values(self):
        border_tiles = []
        true_vals = []
        s = self.game.display_board
        temp_board = np.ones((self.bsize[0], self.bsize[1])) * -1
        for y in range(self.bsize[0]):
            for x in range(self.bsize[1]):
                if 0 < s[y, x] <= 8:
                    n = self.closed_tiles_around(y, x)
                    m = self.flags_around(y, x)
                    for i_inc in range(-1, 2):
                        for j_inc in range(-1, 2):
                            y_new = y + i_inc
                            x_new = x + j_inc
                            if (i_inc != 0 or j_inc != 0) and self.is_in_range(y_new, x_new) and s[y_new, x_new] == COVERED:
                                temp_board[y_new, x_new] = max(temp_board[y_new, x_new], float(s[y, x] - m)/n)
        for y in range(self.bsize[0]):
            for x in range(self.bsize[1]):
                if temp_board[y, x] >= 0:
                    border_tiles.append((y, x))
                    true_vals.append(1 - temp_board[y, x])
        return border_tiles, true_vals

    def all_uncovered(self):
        res = []
        s = self.game.display_board
        for y in range(self.bsize[0]):
            for x in range(self.bsize[1]):
                if s[y, x] == COVERED:
                    res.append((y,x))
        return res

    def make_states_improved(self, border):
        states = np.ones((len(border), 5, 5)) * -1
        s = self.game.display_board + 1
        for k in range(len(border)):
            h, w = border[k]
            for i in range(0, 5):
                h_new = h + i - 2
                for j in range(0, 5):
                    w_new = w + j - 2
                    if h_new >= 0 and w_new >= 0:
                        try:
                            states[k, i, j] = s[h_new, w_new]
                        except:
                            pass
        return states



    def make_states(self, border):
        states = np.zeros((len(border), 5, 5))
        s = self.game.display_board + 1
        for k in range(len(border)):
            h, w = border[k]
            for i in range(0, 5):
                h_new = h + i - 2
                for j in range(0, 5):
                    w_new = w + j - 2
                    if h_new >= 0 and w_new >= 0:
                        try:
                            states[k, i, j] = s[h_new, w_new]
                        except:
                            pass
        return states

    def flags_around(self, y, x):
        n = 0
        s = self.game.display_board
        for i_inc in range(-1, 2):
            for j_inc in range(-1, 2):
                y_new = y + i_inc
                x_new = x + j_inc
                if (i_inc != 0 or j_inc != 0) and self.is_in_range(y_new, x_new) and s[y_new, x_new] == FLAG:
                    n += 1
        return n

    def closed_tiles_around(self, y, x):
        n = 0
        s = self.game.display_board
        for i_inc in range(-1, 2):
            for j_inc in range(-1, 2):
                y_new = y + i_inc
                x_new = x + j_inc
                if (i_inc != 0 or j_inc != 0) and self.is_in_range(y_new, x_new) and s[y_new, x_new] == COVERED:
                    n += 1
        return n

    def place_flags_around(self, y, x):
        s = self.game.display_board
        for i_inc in range(-1, 2):
            for j_inc in range(-1, 2):
                y_new = y + i_inc
                x_new = x + j_inc
                if (i_inc != 0 or j_inc != 0) and self.is_in_range(y_new, x_new) and s[y_new, x_new] == COVERED:
                    self.game.mark((y_new, x_new))

    def place_flags(self):
        h = self.bsize[0]
        w = self.bsize[1]
        s = self.game.display_board
        for y in range(h):
            for x in range(w):
                if 0 < s[y,x] <= 8:
                    if self.closed_tiles_around(y, x) + self.flags_around(y, x) == s[y,x]:
                        self.place_flags_around(y, x)

    def softmax(self, y):
        """ simple helper function here that takes unnormalized logprobs """
        #maxy = np.amax(y)
        #e = np.exp(y - maxy)
        return y / np.sum(y)

    def play(self):
        wt = False
        for i in range(self.num_episodes):
            self.game.reset()
            if (self.game.wins + self.game.losses != i):
                break;
            _, _, d = self.game.open((0, 0))
            while (not d):
                self.place_flags()
                # pmines, true_vals = self.choose_possible_mines()
                pmines, true_vals = self.border_tiles_with_true_values()
                # print pmines
                # print true_vals

                if (len(pmines) == 0):
                    uncovered_list = self.all_uncovered()
                    # print uncovered_list
                    a = random.sample(uncovered_list, 1)[0]
                else:
                    states = self.make_states(pmines)
                    states.shape = (len(pmines), 1, 25)

                    pboard = np.zeros(self.bsize)

                    for i in range(len(pmines)):
                        tmp_state = states[i].flatten()
                        tmp_state = self.toHot(tmp_state)
                        h, w = pmines[i]
                        pboard[h, w] = self.nn.predict(tmp_state)

                    # print pmines
                    # print states
                    # print pboard

                    act = np.argmax(pboard.flatten())
                    a = (act // self.bsize[1], act % self.bsize[1])

                # print a
                _, _, d = self.game.open(a)

                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_SPACE:
                            wt = not wt
                        elif event.key == K_ESCAPE:
                            exit()

                if wt:
                    event = pygame.event.wait()
                    while (1):
                        if event.type == KEYDOWN:
                            if event.key == K_SPACE:
                                wt = not wt
                                break
                            elif event.key == K_RETURN:
                                break
                            elif event.key == K_ESCAPE:
                                exit()
                        event = pygame.event.wait()