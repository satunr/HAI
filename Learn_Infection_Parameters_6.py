# Learns individual infeciton parameters based on sampling individuals for testing.
# Experiment corresponding to figure 6

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


def seird(rank, A, L, S, prob, gamma, CPs, CPuse, delta, gridLoc, CP_zone_, iA, iB, dist=1.8288):
    # Update CP and infectivity of individuals based on new locations
    global T, LogX, LogY, zoneOfChoice

    CPs1 = deepcopy(CPs)
    CPs2 = deepcopy(CPuse)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(L.keys()))

    L = getLoc(Grid, L, A, gridLoc)

    for i in range(len(l) - 1):
        for j in range(i + 1, len(l)):
            if euclidean(L[l[i]], L[l[j]]) < dist:
                neighbors[l[i]].append(l[j])
                neighbors[l[j]].append(l[i])
    
        numInGrid = np.zeros(Grid_X*Grid_Y)
    for person in l:
            numInGrid[gridLoc[person]] += 1
    denseInGrid = [(((math.pi * pow(dist,2) * numInGrid[i])/(dx*dy))-1) for i in range(Grid_X*Grid_Y)]

    for person in l:
        neighbor = neighbors[person]
        if S[person] == 'I':
            if random.uniform(0, 1) < gamma:
                S[person] = 'R'
                CPs1[person] = 0
                CPs2[person] = 0
        elif S[person] == 'R':
            if random.uniform(0, 1) < delta:
                S[person] = 'S'
        elif S[person] == 'S':
            flag = False
            for other in neighbor:
                if random.uniform(0, 1) < iB[person] * CPs[other]:
                    S[person] = 'I'
                    CPs1[person] = 1.0
                    break
            CPs1[person] = iA[person] * CPs[person] + prob * sum([CPs[other] for other in neighbor])
            CPs1[person] = min(CPs1[person], 1.0)
            
            if T>0: 
                CPs2[person] = iA[i] * CPuse[person] + iB[i]* CP_zone_[gridLoc[person]] *(N*rank[gridLoc[person]])
            else: 
                CPs2[person] = CPuse[person]
            CPs2[person] = min(CPs2[person], 1.0)

    # Find the number of inter-zone transitions
    # and estimate PREDICTED zonal CP
    gridLocNew = find_grid(Grid, L)
    W = np.zeros((len(Grid.keys()), len(Grid.keys())))
    for person in gridLoc.keys():
        u = gridLoc[person]
        v = gridLocNew[person]
        W[u, v] += 1

    CP_zone_pred_ = None
    if CP_zone_ is not None:
        CP_zone_pred_ = {g: np.sum([N*rank[gridLoc[g1]]*A[[g1, g]]*CP_zone_[g1] for g1 in Grid.keys()]) / np.sum([N*rank[gridLoc[g1]]*A[[g1, g]] for g1 in Grid.keys()]) for g in Grid.keys()}

    # Estimate REAL zonal CP
    CP_zone_ = {g: np.mean([CPs1[person] for person in CPs1.keys() if gridLocNew[person] == g]) for g in Grid.keys()}

    # Save data for zone of interest in LogX, LogY
    if CP_zone_pred_ is not None:
        LogX.append(CP_zone_[zoneOfChoice])
        LogY.append(CP_zone_pred_[zoneOfChoice])

    return S, CPs1, CPs2, L, gridLocNew, CP_zone_, CP_zone_pred_


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


def test(N):
    # Samples individuals for infection testing
    testnum = int(0.2*N)
    totest = np.zeros(testnum)
    k = 0
    while k<testnum:
        hold = random.randint(1,N)
        if hold not in totest:
            totest[k] = hold
            k += 1
    for i in range(testnum):
        totest[i] = int(totest[i]-1)
    return totest


def learnParams(N, infInfo, cps, iA, iB, lr):
    # Updates individual parameters based on infection data
    global holddiff, T
    testing = test(N)
    w = 0.2
    gam = 0.1
    rate = lr
    inA, inB = iA, iB
    diff = []
    for i in range(len(testing)):
        if infInfo[int(testing[i])] == 'I':
            diff.append(1-cps[int(testing[i])])
            inHoldDiff[int(testing[i])].append(1-cps[int(testing[i])])
            inHoldDiff[N+int(testing[i])].append(T)
    if diff != []:
        avgdiff = np.mean(diff)
        stdiff = np.std(diff)
        holddiff.append(avgdiff)
        holdstd.append(stdiff)
        for i in range(len(testing)):
            if infInfo[int(testing[i])] == 'I':
                d = 1-cps[int(testing[i])]
                inA[int(testing[i])] = inA[int(testing[i])] + rate[int(testing[i])] * (1 - w*d - (1-w)*avgdiff)
                inB[int(testing[i])] = inB[int(testing[i])] + rate[int(testing[i])] * (1 - w*d - (1-w)*avgdiff)
                rate[int(testing[i])] = rate[int(testing[i])]*gam
    return inA, inB, avgdiff, rate

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

# Individual parameters
indvBeta, indvAlpha, mix, lr = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
for i in range(N):
    indvBeta[i], indvAlpha[i], lr[i] = round(random.uniform(0.01,2),2), random.uniform(0.05,0.4), random.uniform(0.4,0.9)

status = ['S', 'E', 'I', 'R', 'D']
E = [0.6, 0, 0.4, 0, 0]

Grid_X, Grid_Y = 7,4
dx, dy = X / Grid_X, Y / Grid_Y
pd = float(N) / (X * Y)
K = C/float(dx*dy)

Grid, t = {}, 0
for i in range(Grid_X):
    for j in range(Grid_Y):
        Grid[t] = [i * dx, (i + 1) * dx, j * dy, (j + 1) * dy]
        t += 1

zoneOfChoice, LogX, LogY, truePop = 6, [], [], []
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

S = np.random.choice(status, p = E, size = N).tolist()

CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}
CPuse = {i: 1 if S[i] == 'I' else 0 for i in range(N)}

# Run simulation
# ====================
T = 0
L = getLoc(Grid, None, A, None)
gridLoc = find_grid(Grid, L)
holddiff,holdstd = [],[]
inHoldDiff = [[] for i in range(2*N)]


CP_zone_ = None
numinf = np.zeros(N)
risk = np.ones(N)
riskAB = np.zeros(N)
while T <= duration:
    oldS = deepcopy(S)
    S, CPs, CPuse, L, gridLoc, CP_zone_, CP_zone_pred_= seird(rank, A, L, S, prob, gamma, CPs, CPuse, delta, gridLoc, CP_zone_, indvAlpha, indvBeta)
    num = getZones(gridLoc, Grid_X, Grid_Y)
    truePop.append(num[zoneOfChoice])
    if T>0: indvAlpha, indvBeta, avgd, lr = learnParams(N, S, CPuse, indvAlpha, indvBeta, lr)
    print(T)
    T += 1


# Graph
# ====================    
plt.plot([i for i in range(len(holddiff))], holddiff, color='red')
plt.fill_between([i for i in range(len(holddiff))], [holddiff[i]-holdstd[i] for i in range(len(holddiff))], [holddiff[i]+holdstd[i] for i in range(len(holddiff))], color='red', alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Average Difference')
plt.title('Average Difference over Time')
plt.tight_layout()
plt.show()
    
