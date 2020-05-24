import concurrent.futures
import numpy as np
import pandas as pd

from Thesis.GenAlg_thesis import GenAlg
from Thesis.SCsettings_thesis import a1, a2, a3, b1, b2, b3, randomArgs, demandSample, Output
from tqdm import tqdm
from Thesis.SupplyChain_thesis import runSC
import time

n_it = 15
T = 1200
lower = 20
upper = 60
tasks = 6
max_gen = 300
chromosomes = []

demand = demandSample(T, lower, upper, n_it, antithetic=True)

results = []
ought = []

iterations = [*range(n_it)]


def it_process(its, demand, arg):
    tscc = []
    for i in tqdm([*range(its)]):
        tscc.append(runSC(arg['opt'], args=arg, demand=demand[i]))
    return tscc


def it_process_wrapper(args):
    return it_process(*args)


def ga_process(its, demand, arg):
    chromosomes = []
    tscc = []
    for i in tqdm([*range(its)]):
        GA = GenAlg(args=arg, demand=demand[i])
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)
    return tscc


def ga_process_wrapper(args):
    return ga_process(*args)


arg = a1
geneticalgorithm = True
start = time.time()

with concurrent.futures.ProcessPoolExecutor() as executor:
    demands = [a.tolist() for a in np.array_split(np.array(demand), tasks)]
    args = ((len(d), d, arg) for d in demands)
    if not geneticalgorithm:
        tscc_split = executor.map(it_process_wrapper, args)
        tscc = []
        for c in tscc_split:
            print(c)
            tscc += c
    else:
        return_split = executor.map(ga_process_wrapper, args)
        tscc = []
        chromosomes = []
        for ret in return_split:
            chromosomes += ret[0]
            tscc += ret[1]

elapsed = time.time() - start

results.append([np.mean(tscc), np.std(tscc)])
ought.append(arg['ought'])

df, ought_df, delta_perc, delta_abs = Output(results, ought, elapsed)
