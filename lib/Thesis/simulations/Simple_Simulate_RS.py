import time
from model.SCsettings_thesis import a1, a2, a3, b1, b2, b3, s1, randomArgs, demandSample, Output
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from model.RandomSearch import RandomSearch
from model.SupplyChain_thesis import runSC


n_it = 2
T = 1200
lower = 20
upper = 60
maxGen = 200

demand = demandSample(T, lower, upper, n_it, antithetic=True)

results = []
tscc = []
ought = []

iterations = [*range(n_it)]

start = time.time()

for i, arg in enumerate([s1]):
    for j in tqdm(iterations):
        RS = RandomSearch(args=arg, demand=demand[i])
        RS.runAlgorithm(maxGen=maxGen)
        tscc.append(RS.tscc)

    results.append([np.mean(tscc), np.std(tscc)])

    ought.append(arg['ought'])

elapsed = time.time() - start
df, ought_df, delta_perc, delta_abs = Output(results, ought, elapsed)