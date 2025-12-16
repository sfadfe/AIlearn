import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import utils as ut ##내가 만든 함수들임...

n = 1000
dot_list = []

temp = set()
while len(temp) < n:
    x = np.random.randint(n//3, 2*n//3)
    y = np.random.randint(n//3, 2*n//3)
    temp.add((x, y))
    
    temp = list(temp)

for i in range(0, 20):
    dot_list.append([temp[i][0], temp[i][1], 1])

for i in range(20, 40):
    dot_list.append([temp[i][0], temp[i][1], 2])

for i in range(40, 60):
    dot_list.append([temp[i][0], temp[i][1], 3])

for k in range(20):
    for i in range(n+1):
        for j in range(n+1):
            if (i,j) not in temp:
                point = [i,j]
                classified = ut.classification(point, dot_list, k, p=2)
                
                if classified == 1:
                    plt.scatter(i, j, color='red', s=0.1)
                elif classified == 2:
                    plt.scatter(i, j, color='blue', s=0.1)
                else:
                    plt.scatter(i, j, color='green', s=0.1)