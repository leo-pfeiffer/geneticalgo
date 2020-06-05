import concurrent.futures
import numpy as np
import pandas as pd
from model.GenAlg_thesis import GenAlg
from model.SCsettings_thesis import s1, s2, s3, s4, randomArgs, demandSample
from tqdm import tqdm
import time
import datetime

n_it = 30
T = 1200
lower = 20
upper = 60
tasks = 6
max_gen = 40
chromosomes = []

demand = demandSample(T, lower, upper, n_it, antithetic=True)

results = pd.DataFrame(columns=["MX", "MP", "CR", "TSCC"])

iterations = [*range(n_it)]


mxs = [0.1, 0.2, 0.3, 0.4, 0.5]
mps = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
crs = [0.5, 0.6, 0.7, 0.8, 0.9, 1]


def ga_process(its, demand, arg, mx, mp, cr):
    chromosomes = []
    tscc = []
    for i in tqdm([*range(its)]):
        GA = GenAlg(args=arg, demand=demand[i], mx=mx, mp=mp, cr=cr)
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)
    return chromosomes, tscc


def ga_process_wrapper(args):
    return ga_process(*args)


arg = s4
geneticalgorithm = True

for mx in mxs:
    for mp in mps:
        for cr in crs:
            start = time.time()

            with concurrent.futures.ProcessPoolExecutor() as executor:
                demands = [a.tolist() for a in np.array_split(np.array(demand), tasks)]
                args = ((len(d), d, arg, mx, mp, cr) for d in demands)
                return_split = executor.map(ga_process_wrapper, args)
                tscc = []
                chromosomes = []
                for ret in return_split:
                    chromosomes += ret[0]
                    tscc += ret[1]

            avg_tscc = np.mean(tscc, axis=0)

            tscc = avg_tscc[-1]
            elapsed = time.time() - start

            row = {"MX": mx, "MP": mp, "CR": cr, "TSCC": tscc}
            print(elapsed, "\n", row, "\n")
            print(datetime.datetime.now().strftime("%H:%M:%S"))
            results = results.append(row, ignore_index=True)

            if mp == 0.5:
                results.to_csv("success.csv", header=True, index=True)

results.to_csv("results_S4.csv", header=True, index=True)