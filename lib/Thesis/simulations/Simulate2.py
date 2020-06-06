import concurrent.futures
import numpy as np
import pandas as pd
from model.GenAlg_thesis import GenAlg
from model.SCsettings_thesis import s1, s2, s3, s4, s5, s6, demandSample
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime


def simulate():
    n_it = 8
    T = 1200
    lower = 20
    upper = 60
    max_gen = 10
    chromosomes = []

    mx = 0.2
    mp = 0.7
    cr = 0.8

    demand = demandSample(T, lower, upper, n_it, antithetic=True)

    tscc = []
    for i in range(n_it):
        GA = GenAlg(args=s6, demand=demand[i], mx=mx, mp=mp, cr=cr)
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)

    tscc = pd.DataFrame(tscc).T
    tscc['Mean'] = tscc.apply(np.mean, 1)

    return tscc, chromosomes


def simulate2():
    n_it = 30
    T = 1200
    lower = 20
    upper = 60
    tasks = 6
    max_gen = 200
    mx = 0.2
    mp = 0.7
    cr = 0.8

    arg = s6

    demand = demandSample(T, lower, upper, n_it, antithetic=True)
    tscc = []
    chromosomes = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        demands = [a.tolist() for a in np.array_split(np.array(demand), tasks)]
        args = ((len(d), d, arg, mx, mp, cr, max_gen) for d in demands)
        return_split = executor.map(ga_process_wrapper, args)

        for ret in return_split:
            chromosomes += ret[0]
            tscc += ret[1]

    tscc = pd.DataFrame(tscc).T
    tscc['Mean'] = tscc.apply(np.mean, 1)
    return tscc, chromosomes


def ga_process(its, demand, arg, mx, mp, cr, max_gen):
    chromosomes = []
    tscc = []
    for i in tqdm([*range(its)]):
        GA = GenAlg(args=arg, demand=demand[i], mx=mx, mp=mp, cr=cr, rechenberg=True)
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)
    return chromosomes, tscc


def ga_process_wrapper(args):
    return ga_process(*args)


def plot(tscc, chromosomes):
    fig = plt.figure(figsize=(9, 6), dpi=300)
    fig.tight_layout(pad=0.1)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([*range(len(tscc))], tscc.Mean.values.tolist())[0]
    ax.set_ylabel("TSCC")
    ax.set_xlabel("Generation number")

    best_chrom = chromosomes[np.argmin([x for x in tscc.iloc[-1,]])].tolist()
    text = "Best policy:\nRetailer: {}, Distributer: {},\nManufacturer: {}, " \
           "Supplier: {}\n--------------------------------------------------\nTSCC: {}"
    text = text.format(best_chrom[0], best_chrom[1], best_chrom[2], best_chrom[3], tscc.Mean.values[-1])

    ax.text(len(tscc) - 1, tscc.Mean.values[0], text, fontsize=10, va="top", ha="right")

    plt.savefig("s6.png")


if __name__ == "__main__":
    t = time.time()
    tscc, chromosomes = simulate2()
    plot(tscc, chromosomes)
    print(time.time()-t)
