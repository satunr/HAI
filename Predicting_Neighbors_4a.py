# Compares the predicted number of neighbors  of an individual with the true number of neighbors
# Experiment corresponding to figure 4a

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
import scipy.stats
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


def seird(L, S, beta, prob, gamma, alpha_decay, CPs, delta, gridLoc, CP_zone_, dx, dy, dist=1.8288):
    # Update CP and infectivity of individuals based on new locations
    global T, LogX, LogY, zoneOfChoice, avgNeigh, expNeigh

    CPs1 = deepcopy(CPs)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(L.keys()))

    L = getLoc(Grid, L, A, gridLoc, T)

    for i in range(len(l) - 1):
        for j in range(i + 1, len(l)):
            if euclidean(L[l[i]], L[l[j]]) < dist:
                neighbors[l[i]].append(l[j])
                neighbors[l[j]].append(l[i])

                

    numGrids = [np.sum([1 for person in l if gridLoc[person]==i]) for i in range(Grid_X*Grid_Y)]
    expNeighHold = [max((math.pi * pow(dist,2) * float(numGrids[i]) / float(dx*dy))-1, 0) for i in range(Grid_X*Grid_Y)]
    trueNeigh = np.zeros(Grid_X*Grid_Y)
    neighborsize = np.zeros(N)
    for i in range(N):
        if np.size(neighbors[i])>0: neighborsize[i] = np.size(neighbors[i])
        else: neighborsize[i] = 0
    

    trueNeigh = [np.mean([np.size(neighbors[person]) for person in l if gridLoc[person]==i]) for i in range(Grid_X*Grid_Y)]
    for i in range(Grid_X*Grid_Y):
        if math.isnan(trueNeigh[i]):
            trueNeigh[i]=0

    avgNeigh.append(np.mean([trueNeigh[i] for i in range(Grid_X*Grid_Y) if trueNeigh[i]>0])) 
    expNeigh.append(np.mean([expNeighHold[i] for i in range(Grid_X*Grid_Y) if expNeighHold[i]>0]))

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
    gridLocNew = find_grid(Grid, L)
    W = np.zeros((len(Grid.keys()), len(Grid.keys())))
    for person in gridLoc.keys():
        u = gridLoc[person]
        v = gridLocNew[person]
        W[u, v] += 1

    CP_zone_pred_ = None
    if CP_zone_ is not None:
        CP_zone_pred_={}
        CP_zone_pred_ = {g: np.sum([W[g1, g] * CP_zone_[g1] for g1 in Grid.keys()]) / np.sum(W[:, g]) for g in Grid.keys()}
        for g in Grid.keys():
            if math.isnan(CP_zone_pred_[g]):CP_zone_pred_[g]=0
    # Estimate REAL zonal CP
    CP_zone_ = {g: np.mean([CPs1[person] for person in CPs1.keys() if gridLocNew[person] == g]) for g in Grid.keys()}

    # Save data for zone of interest in LogX, LogY
    if CP_zone_pred_ is not None:
        LogX.append(CP_zone_[zoneOfChoice])
        LogY.append(CP_zone_pred_[zoneOfChoice])

    return S, CPs1, L, gridLocNew, CP_zone_


def bounds(loc, X, Y):
    # Ensures updated locations fall within the specified bounds

    loc_updated = (np.clip(loc[0], 0, X), np.clip(loc[1], 0, Y))

    return loc_updated


def getLoc(G, L, A, gridLoc):
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


# Initialize variables
# ====================
X, Y = 70,40
N = 330
duration, T = 50, 0

R0 = 3.2
sigma, gamma, delta = 0.25, 0.05, 0.025
beta = gamma * R0

r = 1.8288
C = math.pi * math.pow(r, 2) * (float(N) / float(X * Y))
alpha_decay, prob = 1 / float(np.sqrt(10)), (math.pi * (r ** 2) * N) / (X * Y)

status = ['S', 'E', 'I', 'R', 'D']
E = [0.8, 0, 0.2, 0, 0]

Grid_X, Grid_Y = 7,4
dx, dy = X / Grid_X, Y / Grid_Y
pd = float(N) / (X * Y)

Grid, t = {}, 0
for i in range(Grid_X):
    for j in range(Grid_Y):
        Grid[t] = [i * dx, (i + 1) * dx, j * dy, (j + 1) * dy]
        t += 1

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

zoneOfChoice, LogX, LogY, avgNeigh, expNeigh = 0, [], [], [], []
S = np.random.choice(status, p = E, size = N).tolist()

CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}

# Run simulation
# ====================
T = 0
L = getLoc(Grid, None, A, None, T)
gridLoc = find_grid(Grid, L)

CP_zone_ = None
while T <= duration:
    S, CPs, L, gridLoc, CP_zone_ = seird(L, S, beta, prob, gamma, alpha_decay, CPs, delta, gridLoc, CP_zone_, dx,dy)
    T += 1


# Graph
# ====================
res = scipy.stats.ttest_rel(avgNeigh, expNeigh, axis=0, nan_policy='propagate')
print(res)
av, ex = np.array(avgNeigh), np.array(expNeigh)

deg = 3
a, residuals, rank, singular_values, rcond= np.polyfit(av, ex , deg, full=True)
print(a)
poly = np.poly1d(a)
x = np.linspace(np.min(av), np.max(av), 100)
plt.plot(x, poly(x),color='red', linestyle='-', linewidth=1)
plt.scatter(avgNeigh, expNeigh, color='blue')
plt.xlabel('True Number of Neighbors')
plt.ylabel('Predicted Number of Neighors')
plt.title('True vs Predicted Average Number of Neighbors')
plt.show()
