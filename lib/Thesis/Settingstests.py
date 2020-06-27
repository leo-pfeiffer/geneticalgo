import numpy as np
from SCsettings_paper import demandSample
import time
import random
from tqdm import tqdm
from model.SupplyChain2 import runSC
import pandas as pd


random.seed(123)
np.random.seed(123)


def simulate(arg, name):
    t = time.time()
    tscc_list = []

    for i in tqdm([*range(n_it)]):
        tscc = runSC(chromosome=[100, 100, 100, 100], args=arg, name=name)
        tscc_list.append(tscc)

    tscc_df = pd.DataFrame(tscc_list)
    tscc_df.to_csv(name+'.csv')
    print('\n', name, 'Done. Time:', time.time() - t, '\n\n')


if __name__ == "__main__":
    A = {
        'hcs': np.array([4, 3, 2, 1]),
        'scs': np.array([8, 6, 4, 2]),
        'rlt': np.array([2, 3, 4, 5]),
        'ilt': np.random.randint(0, 11, 4),
        'RMSilt': np.random.randint(0, 11, 1)[0]
    }

    B = {
        'hcs': np.array([6, 5, 3, 1]),
        'scs': np.array([5, 4, 3, 2]),
        'rlt': np.random.randint(0, 11, 4),
        'ilt': np.random.randint(0, 11, 4),
        'RMSilt': np.random.randint(0, 11, 1)[0]
    }

    n_it = 100   # 10000
    T = 1200
    lower = 20
    upper = 60

    demand = demandSample(T, lower, upper, n_it, antithetic=True)

    simulate(arg=A, name='A')
    simulate(arg=B, name='B')