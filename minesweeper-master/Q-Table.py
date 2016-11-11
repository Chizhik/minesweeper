import numpy as np
from game import *

g = Game(4, 4, 4)

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .85
y = .99
num_episodes = 20
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = g.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 20:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,16)*(1./(i+1)))
        b = (a[0]//4, a[0]%4)
        #Get new state and reward from environment
        s1,r,lose = g.open(b)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d != None:
            break
    #jList.append(j)
    rList.append(rAll)