from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras import backend as K
import keras.callbacks
import numpy as np


class NN:
    def __init__(self, bsize = (4, 4), gamma = 0.99):
        self.model = Sequential()
        self.model.add(Convolution2D(nb_filter=bsize[0]*bsize[1], nb_row=3, nb_col=3, activation='relu', border_mode='same', input_shape=(1, bsize[0], bsize[1]), init='uniform'))
        self.model.add(Flatten())
        self.model.add(Dense(bsize[0]*bsize[1], activation='relu', init='uniform'))
        self.model.add(Dense(bsize[0]*bsize[1], activation='linear', init='uniform'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()
        self.tb = keras.callbacks.TensorBoard(log_dir='/tmp/NN/logs', write_graph=True)
        self.bsize = bsize
        self.gamma = gamma

    def predict(self, state):
        test = np.asarray([[state]])
        q = self.model.predict(test)
        action = np.argmax(q, 1)
        return q, action

    def train(self, x, y):
        self.model.train_on_batch(x, y)

    def save(self, fpath = 'model.hdf'):
        self.model.save_weights(fpath)

    def load(self, fpath = 'model.hdf'):
        self.model.load_weights(fpath, by_name = False)