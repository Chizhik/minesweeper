import numpy as np
import tensorflow as tf
from game import *

tf.reset_default_graph()\

n_nodes_hl1 = 16
n_nodes_hl2 = 16
n_nodes_hl3 = 16

inputs1 = tf.placeholder(shape=[1, 16],dtype=tf.float32)

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([16, n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, 16])),'biases':tf.Variable(tf.random_normal([16]))}
	
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.sigmoid(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output    

def train_neural_network():
	Qout = neural_network_model(inputs1)
	predict = tf.argmax(Qout, 1)
	nextQ = tf.placeholder(shape=[1,16],dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	updateModel = trainer.minimize(loss)

	g = Game(4, 4, 4)

	init = tf.initialize_all_variables()

	# Set learning parameters
	y = .99
	e = 0.1
	num_episodes = 2000

	#create lists to contain total rewards and steps per episode
	jList = []
	rList = []
	with tf.Session() as sess:
	    sess.run(init)
	    for i in range(num_episodes):
	        #Reset environment and get first new observation
	        s = g.reset()
	        rAll = 0
	        j = 0
	        #The Q-Network
	        while j < 20:
	            j+=1
	            #Choose an action by greedily (with e chance of random action) from the Q-network
	            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s})
	            print allQ
	            if np.random.rand(1) < e or i < 500:
	                a[0] = np.random.randint(16)
	            a_pos = (a[0]//4, a[0]%4)
	            print a_pos
	            #Get new state and reward from environment
	            s1,r,done = g.open(a_pos)
	            #Obtain the Q' values by feeding the new state through our network
	            Q1 = sess.run(Qout,feed_dict={inputs1:s1})
	            #Obtain maxQ' and set our target value for chosen action.
	            maxQ1 = np.max(Q1)
	            targetQ = allQ
	            targetQ[0,a[0]] = r + y*maxQ1
	            #Train our network using target and predicted Q values
	            _ = sess.run([updateModel],feed_dict={inputs1:s,nextQ:targetQ})
	            rAll += r
	            s = s1
	            if (done == True) and i >= 500:
	                e = 1./((i/50) + 10)
	                break
	        jList.append(j)
	        rList.append(rAll)
	print "Average reward: " + str(sum(rList)/num_episodes)

train_neural_network()
