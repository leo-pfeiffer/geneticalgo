import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.GenAlg_thesis import GenAlg
from model.SCsettings_thesis import a1, a2, a3, b1, b2, b3, s1, randomArgs, demandSample, Output
from tqdm import tqdm
from model.SupplyChain_thesis import runSC
import time

n_it = 30
T = 1200
lower = 20
upper = 60
tasks = 6
max_gen = 30
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
    return chromosomes, tscc


def ga_process_wrapper(args):
    return ga_process(*args)


arg = s1
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

avg_tscc = np.mean(tscc, axis=0)
sd_tscc = np.std(tscc, axis=0)

tscc = avg_tscc[-1]



# pd.DataFrame([avg_tscc, sd_tscc]).T.to_csv("Report.csv")
"""
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([*range(max_gen)], avg_tscc)[0]
ax.set_ylabel("TSCC")
ax.set_xlabel("Generation number")

best_chrom = chromosomes[np.argmin([x[-1] for x in tscc])].tolist()
text = "Best policy:\nRetailer: {}, Distributer: {},\nManufacturer: {}, Supplier: {}\nTSCC: {}"
text = text.format(best_chrom[0], best_chrom[1], best_chrom[2], best_chrom[3], avg_tscc[-1])

ax.text(max_gen - 1, avg_tscc[0], text, fontsize=10, va="top", ha="right")

# plt.savefig("Report.png")
plt.show()"""