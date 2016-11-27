import matplotlib.pyplot as plt
from numpy import mean

with open("losses/loss1.txt", "r") as myfile:
    losses = []
    #line = myfile.readline()
    #while not line.strip():
    for line in myfile:
        losses.append(float(line.strip()))
        #line = myfile.readline()

    avg_len = len(losses) / 50
    avgs = []
    x = []
    for i in range (avg_len):
        nxt = i * 50;
        x.append(nxt)
        avgs.append(mean(losses[nxt:nxt+50]))

    plt.plot(losses)
    plt.plot(x, avgs, 'r-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.show()
