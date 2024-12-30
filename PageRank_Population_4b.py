# Predict mobility using a PageRank process
# Experiment corresponding to figure 4b

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from numpy import linalg as LA
from numpy.linalg import eig, matrix_power, norm
import random

def l1(x):
    # Returns l1-norm
    return np.sum(np.abs(x))

# Initialize variables
# ====================
Gx, Gy=4, 7
A=np.zeros((Gx*Gy,Gx*Gy))
for i in range(Gx*Gy): 
    if i%7<=1: # PR
        for j in range(Gx*Gy):
            if j%7<=1: A[i][j] = random.uniform(0,1) # ->PR
    if i%7>1 and i<14: # WR
        for j in range(Gx*Gy):
            if i%7>1 and i<14: A[i][j] = random.uniform(0.5,1) # ->WR
            if j==16: A[i][j] = random.uniform(0.05,0.2) # ->TR
    if i%7>1 and i>16: # HW
        for j in range(Gx*Gy):
            if i%7>1 and i>16: A[i][j] = random.uniform(0,1) # ->HW
    if i==16: # TR
        for j in range(Gx*Gy):
            if i==16: A[i][j] = random.uniform(0,1) # ->TR
            if i%7>1 and i<14: A[i][j] = random.uniform(0.3, 0.7) # ->WR
            if i%7<=1: A[i][j] = random.uniform(0.3, 0.7) # ->PR
            if i%7>1 and i>16: A[i][j] = random.uniform(0.05, 0.2) # ->HW
for i in range(Gx*Gy):
    A[i] = (1/np.sum(A[i]))*A[i]

G = nx.from_numpy_array(A, create_using = nx.MultiDiGraph())
Pr = nx.pagerank(G, weight = 'weight', alpha = 0.975)

N, T, Loc = 330, 1000, {}

# Place people at random location
for i in range(N):
    Loc[i] = np.random.choice([j for j in range(A.shape[0])], size = 1)[0]

X, Y, W = [], [], 10

# Run simulation
# ====================
for t in range(T):
    print(t)
    print ('Time %d' % t)

    # Move people
    for i in range(N):
        Loc[i] = np.random.choice([j for j in range(A.shape[0])], size = 1, p = A[Loc[i]])[0]

    X.append(Pr[15]*N)
    Y.append(len([j for j in range(N) if Loc[j] == 15]))


# Graph
# ====================
plt.plot([i for i in range(len(X))], X, label = 'Predicted population', color='red')
plt.plot([i for i in range(len(X))], [np.median(Y[i - W: i]) for i in range(len(X))], label = 'True population', color='blue')
plt.title('True and Predicted Population over Time')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend() 
plt.show()
