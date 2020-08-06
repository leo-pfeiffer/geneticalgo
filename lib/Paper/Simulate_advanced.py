# random lead times

import numpy as np
import pandas as pd
from tqdm import tqdm
from model.GenAlg_advanced import GenAlg
from model.SCsettings_paper import a1, a2, a3, b1, b2, b3, advanced_A, advanced_B, \
    randomArgs, demandSample, Output, randomLTlist


def run2(args, name):
    """This allows for nonzero lead times"""
    T = 1200
    tscc = []
    max_gen = 500
    chromosomes = []
    n_it = 6
    iterations = [*range(n_it)]
    demand = demandSample(1200, 20, 60, n_it, antithetic=True)
    ilt_list = randomLTlist(args['ilt'], T, n_it, ilt=True)
    rlt_list = randomLTlist(args['rlt'], T, n_it, ilt=False)

    for i in tqdm(iterations):
        GA = GenAlg(args=args, ilt_list=ilt_list[i], rlt_list=rlt_list[i], demand=demand[i])
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)

    avg_tscc = np.mean(tscc, axis=0)
    sd_tscc = np.std(tscc, axis=0)

    pd.DataFrame([avg_tscc, sd_tscc]).T.to_csv(f"Report_{name}.csv")
    a = [x[-1] for x in tscc]
    b = [list(x) for x in chromosomes]
    c = [[a[i]] + b[i] for i in range(len(a))]
    pd.DataFrame(c).to_csv(f"Chromosomes_{name}.csv")


argA = {
    'hcs': np.array([4, 3, 2, 1]),
    'scs': np.array([8, 6, 4, 2]),
    'rlt': {'arr': np.array([2, 3, 4, 5]), 'rand': False},
    'ilt': {'lower': 0, 'upper': 4, 'rand': True},
    'RMSilt': [0]
}

argB = {
    'hcs': np.array([4, 3, 2, 1]),
    'scs': np.array([8, 6, 4, 2]),
    'rlt': {'lower': 0, 'upper': 4, 'rand': True},
    'ilt': {'lower': 0, 'upper': 4, 'rand': True},
    'RMSilt': [0]
}

run2(args=argA, name='A')
# run2(args=argB, name='B')
