from pg_reinforce import PolicyGradientREINFORCE
import tensorflow as tf
import numpy as np
from game import *
from collections import deque

size = 8
mines = 8

env_name = 'Minesweeper'
env = Game(size, size, mines)

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.000001, decay=0.9)
writer = tf.train.SummaryWriter("/tmp/{}-experiment-1".format(env_name))

state_dim = size*size*10
num_actions = size*size

def toHot(inputState):
	temp = []
	s = np.array([])
	for i in range(inputState.shape[1]):
		sample = np.zeros(10)
		sample[int(inputState[0,i])] = 1.0
		temp.append(sample)
	return np.asmatrix(np.append(s, temp))

def policy_network(states):
	# define policy neural network
	W1 = tf.get_variable("W1", [state_dim, 1024], initializer=tf.random_normal_initializer())
	b1 = tf.get_variable("b1", [1024], initializer=tf.constant_initializer(0))
	h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
	W2 = tf.get_variable("W2", [1024, num_actions], initializer=tf.random_normal_initializer(stddev=0.1))
	b2 = tf.get_variable("b2", [num_actions], initializer=tf.constant_initializer(0))
	p = tf.matmul(h1, W2) + b2
	return p

pg_reinforce = PolicyGradientREINFORCE(sess, optimizer, policy_network, state_dim, num_actions, summary_writer = writer)

MAX_EPISODES = 10000
MAX_STEPS = 10

episode_history = deque(maxlen=100)
for i_episode in xrange(MAX_EPISODES):
	# initialize
	state = env.reset()
	state, _, _ = env.open((0,0))
	state = toHot(state)
	total_rewards = 0

	done = False

	for t in xrange(MAX_STEPS):
		action = pg_reinforce.sampleAction(state[np.newaxis,:])
		next_state, reward, done = env.open((action//size, action%size))

		total_rewards += reward
		pg_reinforce.storeRollout(state, action, reward)

		
		if done: 
			break
		else:
			state = toHot(next_state)

	pg_reinforce.updateModel()

	episode_history.append(total_rewards)
	mean_rewards = np.mean(episode_history)

	print("Episode {}".format(i_episode))
	print("Finished after {} timesteps".format(t+1))
	print("Reward for this episode: {}".format(total_rewards))
	print("Average reward for last 100 episodes: {}".format(mean_rewards))

