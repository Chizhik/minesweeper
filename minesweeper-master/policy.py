import numpy as np
import tensorflow as tf
from game import *

H = 16 # number of hidden layer neurons
D = 16 # input dimensionality
learning_rate = 0.1
batch_size = 5

tf.reset_default_graph()

# This defines the network as it goes from taking an observation of the environment to 
# giving a probability of chosing to the action
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape = [D, H], initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape = [H, D], initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# From here we define the parts of the network needed for learning a good policy
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, D], name = "input_y")
advantages = tf.placeholder(tf.float32, name = "reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(tf.reduce_sum(tf.mul(input_y, input_y - probability), 1))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

# runnung the agent and game
xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
e = 0.1
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 100
init = tf.initialize_all_variables()

# Launch the graph
g = Game(4, 4, 4)


with tf.Session() as sess:
	rendering = False
	sess.run(init)
	observation = g.reset();

	# Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
	gradBuffer = sess.run(tvars)
	for ix, grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0

	while episode_number <= total_episodes:
		x = np.reshape(observation, [1, D])
		print x
		tfprob = sess.run(probability, feed_dict={observations: x})
		print tfprob
		action = np.argmax(tfprob)
		if np.random.rand(1) < e:
			action = np.random.randint(16)
		print action

		xs. append(x) # observation
		y = np.zeros(D)
		y[action] = 1
		ys.append(y)

		# step the environment and get new measurements
		observation, reward, done = g.open((action//4, action%4))
		reward_sum += reward
		print reward
		drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

		if done:
			print "Done"
			episode_number += 1
			epx = np.vstack(xs)
			epy = np.vstack(ys)
			epr = np.vstack(drs)
			tfp = tfps
			xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

			# size the rewards to be unit normal (helps control the gradient estimator variance)
			normalized_epr = epr - np.mean(epr)
			normalized_epr /= np.std(normalized_epr)
			print normalized_epr

			# Get the gradient for this episode, and save it in the gradBuffer
			tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: normalized_epr})
			#print tGrad
			for ix,grad in enumerate(tGrad):
				gradBuffer[ix] += grad

			# If we have completed enough episodes, then update the policy network with our gradients.
			if episode_number % batch_size == 0:
				sess.run(updateGrads, feed_dict = {W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
				for ix,grad in enumerate(gradBuffer):
					gradBuffer[ix] = grad * 0
				# Give a summary of how well our network is doing for each batch of episodes.
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				print 'Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size)
				reward_sum = 0
			observation = g.reset()
print episode_number,'Episodes completed.'

