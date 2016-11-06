import numpy as np
import tensorflow as tf
from game import *

tf.reset_default_graph()\

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100

inputs1 = tf.placeholder(shape=[1, 16],dtype=tf.float32)

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([16, n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, 16])),'biases':tf.Variable(tf.random_normal([16]))}
	
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output    

def train_neural_network():
	Qout = neural_network_model(inputs1)
	predict = tf.argmax(Qout, 1)
	nextQ = tf.placeholder(shape=[1,16],dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	updateModel = trainer.minimize(loss)

	g = Game(4, 4, 4)

	init = tf.initialize_all_variables()

	# Set learning parameters
	y = .99
	e = 0.1
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
	            _ = sess.run([updateModel],feed_dict={inputs1:s,nextQ:targetQ})
	            rAll += r
	            s = s1
	            if (lose != None):
	                e = 1./((i/50) + 10)
	                break
	        jList.append(j)
	        rList.append(rAll)
	print "Average reward: " + str(sum(rList)/num_episodes)

train_neural_network()
