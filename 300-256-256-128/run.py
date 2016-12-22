from agent import *

if __name__ == "__main__":
	a = Agent(16, 16, 20, 20000, False, False)
	a.learn()
	#a.play()
	
	a = Agent(16, 16, 25, 20000, True, False)
	a.learn()

	a = Agent(16, 16, 30, 20000, True, False)
	a.learn()

	a = Agent(16, 16, 35, 20000, True, False)
	a.learn()

	a = Agent(16, 16, 40, 20000, True, False)
	a.learn()
