# Model true and predicted CP and infectivity based on predicted mobility data between grids
# Experiment corresponding to figure 5b

import random
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
from copy import deepcopy


def find_grid(Grid, L):
    # Identify the grid each person is in based on their coordinates
    gridLoc = {}

    for person in L.keys():
        (x, y) = L[person]

        for g in Grid.keys():
            [xl, xr, yl, yr] = Grid[g]
            if xl <= x <= xr and yl <= y <= yr:
                gridLoc[person] = g

    return gridLoc


def seird(Tran, rank, A, L, S, beta, prob, gamma, alpha_decay, CPs, delta, gridLoc, CP_zone_, dist=1.8288):
    # Update CP and infectivity of individuals based on new locations
    global T, LogX, LogY, zoneOfChoice

    CPs1 = deepcopy(CPs)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(L.keys()))
    
    L = getLoc(Grid, L, A, gridLoc, T)

    for i in range(len(l) - 1):
        for j in range(i + 1, len(l)):
            if euclidean(L[l[i]], L[l[j]]) < dist:
                neighbors[l[i]].append(l[j])
                neighbors[l[j]].append(l[i])

    for person in l:
        neighbor = neighbors[person]
        if S[person] == 'I':
            if random.uniform(0, 1) < gamma:
                S[person] = 'R'
                CPs1[person] = 0
        elif S[person] == 'R':
            if random.uniform(0, 1) < delta:
                S[person] = 'S'
        elif S[person] == 'S':
            flag = False
            for other in neighbor:
                if random.uniform(0, 1) < beta * CPs[other]:
                    S[person] = 'I'
                    CPs1[person] = 1.0
                    break
            CPs1[person] = alpha_decay * CPs[person] + prob * sum([CPs[other] for other in neighbor])
            CPs1[person] = min(CPs1[person], 1.0)

    # Find the number of inter-zone transitions
    # and estimate PREDICTED zonal CP
    gridLocNew = find_grid(Grid, L)

    # Estimate zonal CP
    for i in range(Grid_X*Grid_Y): 
        if sum([1 for person in CPs1.keys() if gridLocNew[person] == i])>0:
            CP_zone_[i]=np.mean([CPs1[person] for person in CPs1.keys() if gridLocNew[person] == i])
        else: CP_zone_[i]=0
    if T>0: 
        CP_zone_pred_ = np.zeros(Grid_X*Grid_Y)
        hold = np.matmul(CP_zone_,Tran)
        for i in range(Grid_X*Grid_Y):
            CP_zone_pred_[i] = min(hold[i],1)
    elif T==0: CP_zone_pred_ = CP_zone_

    # Save data for zone of interest in LogX, LogY
    LogX.append(CP_zone_[zoneOfChoice])
    LogY.append(CP_zone_pred_[zoneOfChoice])
    Inf.append(S.count('I'))

    return S, CPs1, L, gridLocNew, CP_zone_, CP_zone_pred_


def bounds(loc, X, Y):
    # Ensures updated locations fall within the specified bounds
    loc_updated = (np.clip(loc[0], 0, X), np.clip(loc[1], 0, Y))

    return loc_updated


def getLoc(G, L, A, gridLoc, T, dist = 3):
    # Finds a new location for each individual
    global X, Y, N
    new_L = {}

    if L is None:
        for i in range(N):
            # start in WR
            new_L[i] = (random.uniform(20, X), random.uniform(0, 20))
    else:
        for i in range(N):
            g = np.random.choice([j for j in range(A.shape[0])], size = 1, p = A[gridLoc[i]])[0]
            loc = (random.uniform(G[g][0], G[g][1]),random.uniform(G[g][2], G[g][3]))
            new_L[i] = bounds(loc, X, Y)

    return new_L


def getPR(A):
    # Generate PageRank for grids using the transition matrix
    G = nx.from_numpy_array(A, create_using = nx.MultiDiGraph())
    Pr = nx.pagerank(G, weight = 'weight', alpha = 0.975)
    return Pr


def getZones(Locs, x, y):
    # Calculate the number of people in each grid
    zonecounts = np.zeros(x*y)
    for i in range(len(Locs)):
        zonecounts[Locs[i]] +=1
    return zonecounts

# Initialize variables
# ====================
X, Y = 70,40
N = 330
duration, T = 100, 0

R0 = 3.2
sigma, gamma, delta = 0.25, 0.05, 0.025
beta = gamma * R0

r = 1.8288
C = math.pi * math.pow(r, 2)
alpha_decay, prob = 1 / float(np.sqrt(10)), (math.pi * (r ** 2) * N) / (X * Y)

status = ['S', 'E', 'I', 'R', 'D']
E = [0.8, 0, 0.2, 0, 0]

Grid_X, Grid_Y = 7,4
dx, dy = X / Grid_X, Y / Grid_Y
pd = float(N) / (X * Y)
K = C/float(dx*dy)

Grid, t = {}, 0
for i in range(Grid_X):
    for j in range(Grid_Y):
        Grid[t] = [i * dx, (i + 1) * dx, j * dy, (j + 1) * dy]
        t += 1

zoneOfChoice, LogX, LogY, truePop, Inf = 1, [], [], [], []

A=np.zeros((Grid_X*Grid_Y,Grid_X*Grid_Y))
for i in range(Grid_X*Grid_Y): 
    if i%7<=1: # PR
        for j in range(Grid_X*Grid_Y):
            if j%7<=1: A[i][j] = random.uniform(0,1) # ->PR
    if i%7>1 and i<14: # WR
        for j in range(Grid_X*Grid_Y):
            if i%7>1 and i<14: A[i][j] = random.uniform(0.5,1) # ->WR
            if j==16: A[i][j] = random.uniform(0.05,0.2) # ->TR
    if i%7>1 and i>16: # HW
        for j in range(Grid_X*Grid_Y):
            if i%7>1 and i>16: A[i][j] = random.uniform(0,1) # ->HW
    if i==16: # TR
        for j in range(Grid_X*Grid_Y):
            if i==16: A[i][j] = random.uniform(0,1) # ->TR
            if i%7>1 and i<14: A[i][j] = random.uniform(0.3, 0.7) # ->WR
            if i%7<=1: A[i][j] = random.uniform(0.3, 0.7) # ->PR
            if i%7>1 and i>16: A[i][j] = random.uniform(0.05, 0.2) # ->HW
for i in range(Grid_X*Grid_Y):
    A[i] = (1/np.sum(A[i]))*A[i]

rank = getPR(A)

Tran = np.zeros((Grid_X*Grid_Y,Grid_X*Grid_Y))
for i in range(Grid_X*Grid_Y):
    for j in range(Grid_X*Grid_Y):
        Tran[i][j] = (rank[i]*N*A[i][j])/np.sum([rank[i]*N*A[i][j] for j in range(Grid_X*Grid_Y)])

S = np.random.choice(status, p = E, size = N).tolist()

CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}

# Run simulation
# ====================
T = 0
L = getLoc(Grid, None, A, None, T)
gridLoc = find_grid(Grid, L)

CP_zone_ = np.zeros(Grid_X*Grid_Y)
CP_zone_pred_ = np.zeros(Grid_X*Grid_Y)
while T <= duration:
    oldS = deepcopy(S)
    S, CPs, L, gridLoc, CP_zone_, CP_zone_pred_= seird(Tran, rank, A, L, S, beta, prob, gamma, alpha_decay, CPs, delta, gridLoc, CP_zone_)
    num = getZones(gridLoc, Grid_X, Grid_Y)
    truePop.append(num[zoneOfChoice])
    print(T)
    T += 1



# Graph
# ====================
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot([i for i in range(len(LogX))], LogX, label = 'True CP', color = 'blue')
ax1.plot([i for i in range(len(LogY))], LogY, label = 'Predicted CP', color = 'green')
ax1.set_ylabel('Mean CP')

ax2.plot([i for i in range(len(Inf))], Inf, label = 'Infected Count', color = 'red')
ax2.set_ylabel('Number Infected')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.title('Mean Population CP and Infected Count over Time')
ax1.set_xlabel('Time')
plt.show()
