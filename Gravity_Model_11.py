# Compares the impact of periodic greedy and optimized assignments on infection spread with a gravity mobility model
# Experiment corresponding to figures 11

import random
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
from copy import deepcopy
from gekko import GEKKO

def makeA(N, Grid_X, Grid_Y, gridLoc):
    # Calculate transition matrix for gravity model
    k = -0.08
    al = 1.78
    #Population per grid
    gridpops = np.zeros(Grid_X*Grid_Y)
    for i in range(N):
        gridpops[gridLoc[i]]+=1
    #Distance between grids
    d = np.zeros((Grid_X*Grid_Y,Grid_X*Grid_Y))
    holdset = [[10*i,10*j]for i in range(Grid_X) for j in range(Grid_Y)]
    for i in range(Grid_X*Grid_Y):
        for j in range(Grid_X*Grid_Y):
            d[i][j] = euclidean(holdset[i], holdset[j])
    # Calculate A
    A=np.zeros((Grid_X*Grid_Y,Grid_X*Grid_Y))
    for i in range(Grid_X*Grid_Y):
        for j in range(Grid_X*Grid_Y):
            A[i][j] = (k*gridpops[i]*gridpops[j]/((d[i][j]+1)**al))-1
    for i in range(Grid_X*Grid_Y):
        if np.sum(A[i])==0:
            A[i]= 0*A[i]
        else:
            A[i] = (1/np.sum(A[i]))*A[i]
    return A


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


def seird(rank, A, L, S, prob, gamma, CPs, delta, gridLoc, CP_zone_, iA, iB, R_g, dist=1.8288):
    # Update CP and infectivity of individuals based on new locations for standard assignment
    global T, LogX, LogY, zoneOfChoice, check

    CPs1 = deepcopy(CPs)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(L.keys()))

    L , R_g= getLoc(Grid, L, A, gridLoc, R_g)

    for i in range(len(l) - 1):
        for j in range(i + 1, len(l)):
            if euclidean(L[l[i]], L[l[j]]) < dist:
                neighbors[l[i]].append(l[j])
                neighbors[l[j]].append(l[i])
    
        numInGrid = np.zeros(Grid_X*Grid_Y)
    for person in l:
            numInGrid[gridLoc[person]] += 1

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
            for other in neighbor:
                if random.uniform(0, 1) < iB[person] * CPs[other]:
                    S[person] = 'I'
                    CPs1[person] = 1.0
                    break

            CPs1[person] = iA[person] * CPs[person] + prob * sum([CPs[other] for other in neighbor])
            CPs1[person] = min(CPs1[person], 1.0)

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
    for g in Grid.keys():
        if math.isnan(CP_zone_[g]): CP_zone_[g]=0

    # Save data for zone of interest in LogX, LogY
    if CP_zone_pred_ is not None:
        LogX.append(CP_zone_[zoneOfChoice])
        LogY.append(CP_zone_pred_[zoneOfChoice])

    return S, CPs1, L, gridLocNew, CP_zone_, CP_zone_pred_


def optseird(Xopt,rank, A, Lo, So, prob, gamma, CPopt, delta, gridLocO, CP_zone_O, iA, iB, Ropt, dist=1.8288):
    # Update CP and infectivity of individuals based on new locations for optimized assignment
    global T, Copt, N, check
   
    CPs1 = deepcopy(CPopt)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(Lo.keys()))
    R=Ropt
    if T>0: Lo, R = getLocOpt(Grid, Xopt, CP_zone_O, R)
    else: Lo, R = getLoc(Grid, L, A, gridLoc, R)

    for i in range(len(l) - 1):
        for j in range(i + 1, len(l)):
            if euclidean(L[l[i]], L[l[j]]) < dist:
                neighbors[l[i]].append(l[j])
                neighbors[l[j]].append(l[i])
    
        numInGrid = np.zeros(Grid_X*Grid_Y)
    for person in l:
            numInGrid[gridLocO[person]] += 1

    for person in l:
        neighbor = neighbors[person]
        if So[person] == 'I':
            if random.uniform(0, 1) < gamma:
                So[person] = 'R'
                CPs1[person] = 0
        elif So[person] == 'R':
            if random.uniform(0, 1) < delta:
                So[person] = 'S'
        elif So[person] == 'S':
            for other in neighbor:
                if random.uniform(0, 1) < iB[person] * CPopt[other]:
                    So[person] = 'I'
                    CPs1[person] = 1.0
                    break
            CPs1[person] = iA[person] * CPopt[person] + prob * sum([CPopt[other] for other in neighbor])
            CPs1[person] = min(CPs1[person], 1.0)

    # Find the number of inter-zone transitions
    # and estimate PREDICTED zonal CP
    gridLocNew = find_grid(Grid, Lo)
    W = np.zeros((len(Grid.keys()), len(Grid.keys())))
    for person in gridLocO.keys():
        u = gridLocO[person]
        v = gridLocNew[person]
        W[u, v] += 1

    CP_zone_pred_O = None
    if CP_zone_O is not None:
        CP_zone_pred_O = {g: np.sum([N*rank[gridLocO[g1]]*A[[g1, g]]*CP_zone_O[g1] for g1 in Grid.keys()]) / np.sum([N*rank[gridLocO[g1]]*A[[g1, g]] for g1 in Grid.keys()]) for g in Grid.keys()}

    # Estimate REAL zonal CP
    CP_zone_O = {g: np.mean([CPs1[person] for person in CPs1.keys() if gridLocNew[person] == g]) for g in Grid.keys()}
    for g in Grid.keys():
        if math.isnan(CP_zone_O[g]): CP_zone_O[g]=0
    return So, CPs1, Lo, gridLocNew, CP_zone_O, CP_zone_pred_O, R


def bounds(loc, X, Y):
    # Ensures updated locations fall within the specified bounds
    loc_updated = (np.clip(loc[0], 0, X), np.clip(loc[1], 0, Y))

    return loc_updated


def getLoc(G, L, A, gridLoc, R_g):
    # Finds a new location for each individual basedon greedy assignment
    global X, Y, N
    new_L = {}
    R=R_g

    if L is None:
        for i in range(N):
            new_L[i] = (random.uniform(20, X), random.uniform(0, 20))
    else:
        for i in range(N):
            if np.sum(R[i])!=0:
                for k in range(Grid_X*Grid_Y):
                    if R[i][k]==1:
                        for j in range(Grid_X*Grid_Y):
                            if Copt[j][k]==1:
                                g=j
                                R[i][k]=0
            else: 
                g = np.random.choice([j for j in range(A.shape[0])], size = 1, p = A[gridLoc[i]])[0]
            loc = (random.uniform(G[g][0], G[g][1]),random.uniform(G[g][2], G[g][3]))
            new_L[i] = bounds(loc, X, Y)
    return new_L, R


def getLocOpt(G, Xopt, CP, Ropt):
    # Finds a new location for each individual based on optimized location
    new_L = {}
    R=Ropt
    for i in range(N):
        visit, cps = [], []
        for j in range(Grid_X*Grid_Y): 
            if Xopt[i][j].value[0]==1.0: 
                visit.append(j)
                cps.append(1/(CP[j]+0.01))

        g = np.random.choice(visit)
 
        loc = (random.uniform(G[g][0], G[g][1]),random.uniform(G[g][2], G[g][3]))
        new_L[i] = bounds(loc, X, Y)
        for k in range(Grid_X*Grid_Y):
            if Copt[g][k]==1:
                R[i][k]=0
    return new_L, R


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
        totest[i] = totest[i]-1
    return totest


def learnParams(N, infInfo, cps, zoneCP, gridLoc, iA, iB, lr):
    # Updates individual parameters based on infection data
    global holddiff
    testing = test(N)
    w = 0.2
    gam = 0.1
    rate = lr
    inA, inB = iA, iB
    diff = []
    for i in range(len(testing)):
        if infInfo[i] == 'I':
            diff.append(1-cps[i])
    if diff != []:
        avgdiff = np.mean(diff)
        holddiff.append(avgdiff)
        for i in range(len(testing)):
            if infInfo[i] == 'I':
                d = 1-cps[i]
                inA[i] = inA[i] + rate[i] * (1 - w*d - (1-w)*avgdiff)
                inB[i] = inB[i] + rate[i] * (1 - w*d - (1-w)*avgdiff)
                rate[i] = rate[i]*gam
    else: avgdiff=0 
    return inA, inB, avgdiff, rate


def obj(X,CP_zone_O):
    # Objective funciton for optimization
    global N, Grid_X, Grid_Y, indvAlpha, indvBeta, CPopt, rank
    return m.sum([indvAlpha[j]*CPopt[j] + indvBeta[j]*N*(X[j][i]*CP_zone_O[i])*(X[j][i]*rank[i]) for i in range(Grid_X*Grid_Y) for j in range(N)])


# Initialize variables
# ====================
X, Y = 70, 40 
N = 330
duration, T = 120, 0 
R0 = 3.2
sigma, gamma, delta = 0.25, 0.05, 0.025
beta = gamma * R0
r = 1.8288
C = math.pi * math.pow(r, 2)
alpha_decay, prob = 1 / float(np.sqrt(10)), (math.pi * (r ** 2) * N) / (X * Y)
# Individual parameters
indvBeta, indvAlpha, mix, lr = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
for i in range(N):
    indvBeta[i], indvAlpha[i], lr[i] = round(random.uniform(0.5,2),2), random.uniform(0.05,0.4), random.uniform(0.4,0.9) 

indvBetaOpt, indvAlphaOpt = deepcopy(indvBeta), deepcopy(indvAlpha)
status = ['S', 'E', 'I', 'R', 'D']
E = [0.8, 0, 0.2, 0, 0]
Grid_X, Grid_Y = 7,4
dx, dy = X / Grid_X, Y / Grid_Y
pd = float(N) / (X * Y)
K = C/float(dx*dy)
zoneOfChoice, LogX, LogY, truePop = 2, [], [], []

# Optimization parameters
failures=0
Copt, Ropt = np.eye(Grid_X*Grid_Y), np.zeros((N,Grid_X*Grid_Y)) 
np.random.shuffle(Copt)
for i in range(N):
    hold=random.randint(2,int(Grid_X*Grid_Y/2))
    for k in range(hold):
        Ropt[i][random.randint(0,Grid_X*Grid_Y-1)]=1
R_g = deepcopy(Ropt)

# Initialize locations
Grid, t = {}, 0
for i in range(Grid_X):
    for j in range(Grid_Y):
        Grid[t] = [i * dx, (i + 1) * dx, j * dy, (j + 1) * dy]
        t += 1

S = np.random.choice(status, p = E, size = N).tolist()
So = deepcopy(S)

CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}
CPopt = {i: 1 if S[i] == 'I' else 0 for i in range(N)}
# Initial A
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

# Run simulation
# ====================
T = 0
Tp=0
L, R_g = getLoc(Grid, None, A, None, R_g)
Lo = deepcopy(L)
gridLoc = find_grid(Grid, L)
gridLocO = find_grid(Grid, Lo)
holddiff = []
avgAlpha, avgBeta, alphReal, betaReal = [], [], 1, 1
avgAlphaO, avgBetaO, alphRealO, betaRealO = [], [], 1, 1

index=3*(duration+1)
CPRegTrack, CPOptTrack, InfRegTrack, InfOptTrack = np.zeros(index), np.zeros(index), np.zeros(index), np.zeros(index)

CP_zone_ = None
CP_zone_O = None
while T <= duration:
    print(T)
    A = makeA(N, Grid_X, Grid_Y, gridLoc)
    rank = getPR(A)

    if T>(55) and T<(70): 
        R0 = 9
        beta = gamma * R0
        for i in range(N):
            if T%4==0 and i%((T%13))==0:
                S[i]='I'
                CPs[i]=1
                So[i]='I'
                CPopt[i]=1

    Shold=deepcopy(S) 
    Sohold=deepcopy(So)

    #Model
    if T%8==0:
        check = 0
        hours = 2
        # CP work and parameter learning
        S, CPs, L, gridLoc, CP_zone_, CP_zone_pred_= seird(rank, A, L, S, prob, gamma, CPs, delta, gridLoc, CP_zone_, indvAlpha, indvBeta, R_g)
          
        m = GEKKO(remote=False) # Initialize gekko
        m.options.solver=1
        m.options.MAX_ITER=10000
        m.options.MAX_TIME = 1500 
        # Binary decision variables 
        Xopt = [[m.Var(1, lb = 0, ub = 1, integer = True) for k in range(Grid_X*Grid_Y)] for j in range(N)] 
        if T==0: m.Minimize(obj(Xopt,CP_zone_))
        else: m.Minimize(obj(Xopt,CP_zone_O))
        #Condition 2
        for i in range(N):
            for k in range(Grid_X*Grid_Y):
                if Ropt[i][k] == 1:
                    for j in range(Grid_X*Grid_Y):
                        if Copt[j][k]==1:
                            m.Equation(Xopt[i][j]==1)
            m.Equation(m.sum([Xopt[i][j] for j in range(Grid_X*Grid_Y)])>=1) # Condition 1
        try: 
            m.solve(disp=False)
            if T==0:
                So, CPopt, Lo, gridLocO, CP_zone_O, CP_zone_pred_O, Ropt = optseird(Xopt, rank, A, Lo, So, prob, gamma, CPopt, delta, gridLocO, CP_zone_, indvAlphaOpt, indvBetaOpt, Ropt)
            else:
                So, CPopt, Lo, gridLocO, CP_zone_O, CP_zone_pred_O, Ropt = optseird(Xopt, rank, A, Lo, So, prob, gamma, CPopt, delta, gridLocO, CP_zone_O, indvAlphaOpt, indvBetaOpt, Ropt)
            
        except:
            failures += 1
            So, CPopt, Lo, gridLocO, CP_zone_O, CP_zone_pred_O, Ropt = optseird(Xopt, rank, A, Lo, So, prob, gamma, CPopt, delta, gridLocO, CP_zone_O, indvAlphaOpt, indvBetaOpt, Ropt)
        
        num = getZones(gridLoc, Grid_X, Grid_Y)
        print('Ran on ', T, 'there have been ', failures, ' failures')
        CPRegTrack[Tp], CPOptTrack[Tp] = np.mean([CP_zone_[g] for g in Grid.keys()]), np.mean([CP_zone_O[g] for g in Grid.keys()]) 
        for i in range(N):
            if S[i]=='I':
                InfRegTrack[Tp]+=1 
            if So[i]=='I':
                InfOptTrack[Tp]+=1
        Tp+=1
    else: hours = 3
    truePop.append(num[zoneOfChoice])

    for iter in range(hours):
        S, CPs, L, gridLoc, CP_zone_, CP_zone_pred_= seird(rank, A, L, S, prob, gamma, CPs, delta, gridLoc, CP_zone_, indvAlpha, indvBeta, R_g)
        So, CPopt, Lo, gridLocO, CP_zone_O, CP_zone_pred_O = seird(rank, A, Lo, So, prob, gamma, CPopt, delta, gridLocO, CP_zone_O, indvAlphaOpt, indvBetaOpt, Ropt)
        CPRegTrack[Tp], CPOptTrack[Tp] = np.mean([CP_zone_[g] for g in Grid.keys()]), np.mean([CP_zone_O[g] for g in Grid.keys()]) 
        for i in range(N):
            if S[i]=='I':
                InfRegTrack[Tp]+=1 
            if So[i]=='I':
                InfOptTrack[Tp]+=1
        Tp+=1
            
    if T%3==0 and T>0:
        indvAlpha, indvBeta, avgd, lr = learnParams(N, S, CPs, CP_zone_, gridLoc, indvAlpha, indvBeta, lr)
        indvAlphaOpt, indvBetaOpt, avgdO, lr = learnParams(N, So, CPopt, CP_zone_O, gridLocO, indvAlphaOpt, indvBetaOpt, lr)
        avgAlpha.append(np.mean(indvAlpha))
        avgBeta.append(np.mean(indvBeta))
        avgAlphaO.append(np.mean(indvAlphaOpt))
        avgBetaO.append(np.mean(indvBetaOpt))
        if avgd<0.01: alphReal, betaReal = np.mean(indvAlpha), np.mean(indvBeta)
        if avgdO<0.01: alphRealO, betaRealO = np.mean(indvAlphaOpt), np.mean(indvBetaOpt)
    T += 1

# Graph
# ====================

# Plot CP
plt.plot([i/3 for i in range(len(CPRegTrack))], CPRegTrack, label='Greedy Uniform', color='blue')
plt.plot([i/3 for i in range(len(CPOptTrack))], CPOptTrack, label='Optimization', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Mean Zonal CP')
plt.title('Mean Zonal CP over Time')
plt.show()

# Plot infected population
plt.plot([i/3 for i in range(len(InfRegTrack))], InfRegTrack, label='Greedy Uniform', color='blue')
plt.plot([i/3 for i in range(len(InfOptTrack))], InfOptTrack, label='Optimization', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Infected Number')
plt.title('Infected Number over Time')
plt.show()

