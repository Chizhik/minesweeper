from game import *
from ExperienceBuf import *
from NN import *
import numpy as np

class Agent:
    def __init__(self, load = False):
        self.bsize = (4, 4)
        self.game = Game(4, 4, 3)
        self.nn = NN()
        if load:
            self.nn.load()
        self.buf = ExperienceBuf()
        self.num_episodes = 5000
        self.max_game_moves = 20
        self.observe = 200
        self.e = 0.1
        self.y = 0.99
        self.mini_batch = 32

    def learn(self):
        count = 0

        for i in range(self.num_episodes):
            self.game.reset()
            s, r, d = self.game.open((0, 0))

            while (not d):
                q, act = self.nn.predict(s)
                a = (act[0] // self.bsize[1], act[0] % self.bsize[1])
                if np.random.rand(1) < self.e:
                    a = (np.random.randint(self.bsize[0]), np.random.randint(self.bsize[1]))

                print a
                s1, r, d = self.game.open(a)
                #1, a1 = self.nn.predict(s1)
                #max_q1 = np.max(q1)
                targetQ = q
                targetQ[0, act[0]] = r

                self.buf.memorize(s, targetQ)

                s = s1

                count += 1

                if count > self.observe:
                    x, y = self.buf.get_batch(self.mini_batch)
                    x = np.asarray(x)
                    x.shape = (self.mini_batch, 1, self.bsize[0], self.bsize[1])
                    y = np.asarray(y)
                    y.shape = (self.mini_batch, self.bsize[0] * self.bsize[1])
                    self.nn.train(x, y)

                if count % 50 == 0:
                    self.nn.save()