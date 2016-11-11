import numpy as np
import numpy.random as nrand
import random
import os
import tensorflow as tf

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
        self.display_board = np.zeros((w, h))
        self.display_board += COVERED

        self.board = np.zeros((w, h))

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
        
        return np.matrix(self.display_board.flatten())

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

            else:
                pass # we encountered bomb in the reccursion, don't open the tile0

    def open(self, pos):
        y, x = pos
        temp = self.opened
        reward = 1.0
        
        if self.display_board[y,x] != -1:
            reward = -100.0
        
        elif self.display_board[y,x] == BOMB:
            self.display_board[y,x] = BOOM
            for h in range(self.height):
                for w in range(self.width):
                    if self.display_board[h,w] == COVERED:
                        self.display_board[h,w] = self.board[h][w]
                    elif self.display_board[h,w] == FLAG and self.board[h][w] == BOMB:
                        self.display_board[h,w] = CORRECT
                    elif self.display_board[h,w] == FLAG and self.board[h][w] != BOMB:
                        self.display_board[h,w] = INCORRECT

            reward = -100.0
            self.finished = True
            self.losses += 1
            self.result = False
            
        else:
            self.open_recursive(y, x)
            

        if self.checkWin():
            reward = 1000.0
        if TEST:
            print self.display_board
        return np.matrix(self.display_board.flatten()), reward, self.result

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
        return not self.finished


tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, 16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,16],0,0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,16],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

g = Game(4, 4, 4)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.4
num_episodes = 50
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = g.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 20:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s})
            a = (a[0]//4, a[0]%4)
            if np.random.rand(1) < e:
                a = (np.random.randint(4),np.random.randint(4))
            #Get new state and reward from environment
            print a
            s1,r,lose = g.open(a)
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s1})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:s,nextQ:targetQ})
            rAll += r
            s = s1
            if (lose != None):
                break
        jList.append(j)
        rList.append(rAll)
print "Average reward: " + str(sum(rList)/num_episodes)