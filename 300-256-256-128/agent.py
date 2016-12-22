from game import *
from ExperienceBuf import *
from NN import *
import numpy as np
'''
import matplotlib.pyplot as plt
'''

class Agent:
    def __init__(self, h, w, mines, episodes, load = False, disp = False):
        self.bsize = (h, w)
        self.game = Game(h, w, mines, disp)
        self.nn = NN(self.bsize)
        if load:
            self.nn.load()
        self.buf = ExperienceBuf(400)
        self.num_episodes = episodes
        self.max_game_moves = 64
        self.observe = 10
        self.e = 0.0
        self.y = 0.99
        self.mini_batch = 200

    def toHot(self, inputState):
        temp = []
        s = np.array([])
        for i in range(25):
            sample = np.zeros(12)
            #if (inputState[i] >= 0):
            sample[int(inputState[i])] = 1.0
            temp.append(sample)
        return np.asmatrix(np.append(s, temp))

    def learn(self):
        count = 0
        f = open('losses.txt', 'w')
        for i in range(self.num_episodes):
            self.game.reset()
            _, _, d = self.game.open((0, 0))
            while (not d):
                pmines, true_vals = self.choose_possible_mines()

                states = self.make_states(pmines)
                states.shape = (len(pmines), 1, 25)

                pboard = np.zeros(self.bsize)

                for i in range(len(pmines)):
                    tmp_state = states[i].flatten()
                    tmp_state = self.toHot(tmp_state)
                    prob = self.nn.predict(tmp_state)
                    y_temp, x_temp = pmines[i]
                    if prob < 0.1:
                        self.game.mark(pmines[i])
                        self.buf.memorize(tmp_state, true_vals[i])
                    elif prob > 0.1 and self.game.display_board[y_temp, x_temp] == FLAG:
                        self.game.unmark(pmines[i])


                for i in range(len(pmines)):
                    tmp_state = states[i].flatten()
                    tmp_state = self.toHot(tmp_state)
                    self.buf.memorize(tmp_state, true_vals[i])
                    h, w = pmines[i]
                    pboard[h, w] = self.nn.predict(tmp_state)

                #print pmines
                #print states
                #print pboard

                act = np.argmax(pboard.flatten())
                a = (act // self.bsize[1], act % self.bsize[1])

                if np.random.rand(1) < self.e:
                    a = (np.random.randint(self.bsize[0]), np.random.randint(self.bsize[1]))


                #print a
                _, _, d = self.game.open(a)

                count += 1

            if count > self.observe:
                x, y, bsize = self.buf.get_batch(self.mini_batch)
                x = np.asarray(x)
                x.shape = (bsize, 300)
                y = np.asarray(y)
                y.shape = (bsize, 1)
                self.nn.train(x, y)

            if count % 50 == 0:
                self.nn.save()
                f.write("\n".join([str(e) for e in self.nn.hist.losses]) + "\n")
                self.nn.hist.losses = []
        f.close()
        #self.graph_stats()

    '''
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
    '''

    def make_target(self):
        target = np.zeros(self.bsize)
        s = self.game.display_board
        for h in range(self.bsize[0]):
            for w in range(self.bsize[1]):
                if s[h, w] == COVERED:
                    flag = False
                    for i in range(max(0, h - 1), min(h + 2, self.bsize[0])):
                        for j in range(max(0, w - 1), min(w + 2, self.bsize[1])):
                            if (s[i, j] > 0):
                                flag = True
                                break

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
                if s[h, w] == COVERED or s[h, w] == FLAG:
                    flag = False
                    for i in range(max(0, h - 1), min(h + 2, self.bsize[0])):
                        for j in range(max(0, w - 1), min(w + 2, self.bsize[1])):
                            if (s[i, j] > 0):
                                flag = True
                                break

                        if flag: break

                    if flag:
                        pmines.append((h, w))
                        if self.game.board[h, w] == BOMB: true_vals.append(0)
                        else:true_vals.append(1)
        return pmines, true_vals

    def make_states(self, border):
        states = np.ones((len(border), 5, 5))*(-1)
        s = self.game.display_board + 1
        for k in range(len(border)):
            h, w = border[k]
            for h_new in range(max(0, h - 2), min(h + 3, self.bsize[0])):
                i = h_new - h + 2
                for w_new in range(max(0, w - 2), min(w + 3, self.bsize[1])):
                    j = w_new - w + 2
                    states[k, i, j] = s[h_new, w_new]

        return states


    def softmax(self, y):
        """ simple helper function here that takes unnormalized logprobs """
        #maxy = np.amax(y)
        #e = np.exp(y - maxy)
        return y / np.sum(y)
    '''
    def play(self):
        wt = False
        for i in range(self.num_episodes):
            self.game.reset()
            _, _, d = self.game.open((0, 0))

            while (not d):
                pmines, true_vals = self.choose_possible_mines()

                states = self.make_states(pmines)
                states.shape = (len(pmines), 1, 25)

                pboard = np.zeros(self.bsize)

                for i in range(len(pmines)):
                    tmp_state = states[i].flatten()
                    tmp_state = self.toHot(tmp_state)
                    prob = self.nn.predict(tmp_state)
                    y_temp, x_temp = pmines[i]
                    if prob < 0.1:
                        self.game.mark(pmines[i])
                        #self.buf.memorize(tmp_state, true_vals[i])
                    elif prob > 0.1\
                            and self.game.display_board[y_temp, x_temp] == FLAG:
                        self.game.unmark(pmines[i])

                temp = pygame.Surface((self.bsize[1], self.bsize[0]))
                tran = pygame.Surface((self.game.tileSize - 1, self.game.tileSize - 1))

                for i in range(len(pmines)):
                    tmp_state = states[i].flatten()
                    tmp_state = self.toHot(tmp_state)
                    #self.buf.memorize(tmp_state, true_vals[i])
                    h, w = pmines[i]
                    pboard[h, w] = self.nn.predict(tmp_state)

                    g = 1 - pboard[h, w]
                    g *= 255
                    tran.fill((160, 255-g, g))
                    tran.set_alpha(120)
                    self.game.screen.blit(tran, (w * self.game.tileSize, h * self.game.tileSize))

                pygame.display.flip()


                # print pmines
                # print states
                # print pboard

                act = np.argmax(pboard.flatten())
                a = (act // self.bsize[1], act % self.bsize[1])

                if np.random.rand(1) < self.e:
                    a = (np.random.randint(self.bsize[0]), np.random.randint(self.bsize[1]))

                # print a


                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_SPACE:
                            wt = not wt
                        elif event.key == K_ESCAPE:
                            exit()

                if wt:
                    event = pygame.event.wait()
                    while(1):
                        if event.type == KEYDOWN:
                            if event.key == K_SPACE:
                                wt = not wt
                                break
                            elif event.key == K_RETURN:
                                break
                            elif event.key == K_ESCAPE:
                                exit()
                        event = pygame.event.wait()

                _, _, d = self.game.open(a)

                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_SPACE:
                            wt = not wt
                        elif event.key == K_ESCAPE:
                            exit()

                if wt:
                    event = pygame.event.wait()
                    while(1):
                        if event.type == KEYDOWN:
                            if event.key == K_SPACE:
                                wt = not wt
                                break
                            elif event.key == K_RETURN:
                                break
                            elif event.key == K_ESCAPE:
                                exit()
                        event = pygame.event.wait()
    '''
