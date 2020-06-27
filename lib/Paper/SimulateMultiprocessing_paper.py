import concurrent.futures
import numpy as np
import pandas as pd
from SCsettings_paper import a1, a2, a3, b1, b2, b3, randomArgs, demandSample, Output
from tqdm import tqdm
from SupplyChain_paper import runSC
import time

n_it = 1000
T = 1200
lower = 20
upper = 60
tasks = 6

demand = demandSample(T, lower, upper, n_it, antithetic=True)

results = []
tscc = []
ought = []

iterations = [*range(n_it)]


def it_process(its, demand, arg):
    tscc = []
    for i in tqdm([*range(its)]):
        tscc.append(runSC(arg['opt'], args=arg, demand=demand[i]))
    return tscc


def it_process_wrapper(args):
    return it_process(*args)


arg = a1

start = time.time()

with concurrent.futures.ProcessPoolExecutor() as executor:
    demands = [a.tolist() for a in np.array_split(np.array(demand), tasks)]
    args = ((len(d), d, arg) for d in demands)
    tscc_split = executor.map(it_process_wrapper, args)
    tscc = []
    for c in tscc_split:
        tscc += c

elapsed = time.time() - start

results.append([np.mean(tscc), np.std(tscc)])
ought.append(arg['ought'])

df, ought_df, delta_perc, delta_abs = Output(results, ought, elapsed)