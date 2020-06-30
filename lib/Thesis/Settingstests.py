import numpy as np
import time
from tqdm import tqdm
from model.SupplyChain2 import runSC
import pandas as pd


def simulate(name):
    t = time.time()
    tscc_list = []

    for i in tqdm([*range(n_it)]):
        if name == "A":
            arg = {
                'hcs': np.array([4, 3, 2, 1]),
                'scs': np.array([8, 6, 4, 2]),
                'rlt': np.array([2, 3, 4, 5]),
                'ilt': np.random.randint(0, 5, 4),
                'RMSilt': np.random.randint(0, 5, 1)[0]
            }
        elif name == "B":
            arg = {
                'hcs': np.array([4, 3, 2, 1]),
                'scs': np.array([8, 6, 4, 2]),
                'rlt': np.random.randint(0, 5, 4),
                'ilt': np.random.randint(0, 5, 4),
                'RMSilt': np.random.randint(0, 5, 1)[0]
            }
        tscc = runSC(chromosome=[100, 100, 100, 100], args=arg, name=name)
        tscc_list.append(tscc)

    tscc_df = pd.DataFrame(tscc_list)
    tscc_df.to_csv(name + '2.csv')
    print('\n', name, 'Done. Time:', time.time() - t, '\n\n')


if __name__ == "__main__":
    n_it = 10000  # 10000

    simulate(name='A')
    simulate(name='B')
