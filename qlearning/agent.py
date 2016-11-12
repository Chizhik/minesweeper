from game import *
from ExperienceBuf import *
from NN import *
import numpy as np

class Agent:
    def __init__(self, load = False):
        self.bsize = (4, 4)
        self.game = Game(4, 4, 3)
        self.nn = NN(self.bsize)
        if load:
            self.nn.load()
        self.buf = ExperienceBuf()
        self.num_episodes = 10000
        self.max_game_moves = 20
        self.observe = 200
        self.e = 0.1
        self.y = 0.99
        self.mini_batch = 32

    def toHot(self, inputState):
        temp = []
        s = np.array([])
        for i in range(self.bsize[0] * self.bsize[1]):
            sample = np.zeros(10)
            sample[int(inputState[0, i])] = 1.0
            temp.append(sample)
        return np.asmatrix(np.append(s, temp))

    def learn(self):
        count = 0

        for i in range(self.num_episodes):
            self.game.reset()
            s, r, d = self.game.open((0, 0))
            while (not d):
                s = self.toHot(s.flatten())
                q, act = self.nn.predict(s)
                a = (act[0] // self.bsize[1], act[0] % self.bsize[1])
                if np.random.rand(1) < self.e:
                    a = (np.random.randint(self.bsize[0]), np.random.randint(self.bsize[1]))


                q.shape = (self.bsize[0], self.bsize[1])
                print q
                targetQ = self.make_target()
                print targetQ
                print a
                targetQ.shape = (1, self.bsize[0] * self.bsize[1])
                s1, r, d = self.game.open(a)
                #1, a1 = self.nn.predict(s1)
                #max_q1 = np.max(q1)
                #targetQ = q
                #targetQ[0, act[0]] = r

                self.buf.memorize(s, targetQ)

                s = s1

                count += 1

            if count > self.observe:
                x, y = self.buf.get_batch(self.mini_batch)
                x = np.asarray(x)
                x.shape = (self.mini_batch, self.bsize[0] * self.bsize[1]*10)
                y = np.asarray(y)
                y.shape = (self.mini_batch, self.bsize[0] * self.bsize[1])
                self.nn.train(x, y)

                if count % 50 == 0:
                    self.nn.save()

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
                        if self.game.board[h, w] == BOMB: target[h, w] = 0
                        else: target[h, w] = 1
                    else: target[h, w] = 0.5
        return self.softmax(target)

    def softmax(self, y):
        """ simple helper function here that takes unnormalized logprobs """
        #maxy = np.amax(y)
        #e = np.exp(y - maxy)
        return y / np.sum(y)