import numpy as np
from tqdm import tqdm
from supplychain import runSC
from scsettings import a1, a2, a3, b1, b2, b3, randomArgs, demandSample, Output, randomLTlist


def run():
    args = {
        'hcs': np.array([4, 3, 2, 1]),
        'scs': np.array([8, 6, 4, 2]),
        'rlt': {'arr': np.array([2, 3, 4, 5]), 'rand': False},
        'ilt': {'lower': 0, 'upper': 4, 'rand': True},
        'RMSilt': np.array([0])
    }
    T = 1200
    n_it = 2
    demand = demandSample(1200, 20, 60, n_it, antithetic=True)
    ilt_list = randomLTlist(args['ilt'], T, n_it, ilt=True)
    rlt_list = randomLTlist(args['rlt'], T, n_it, ilt=False)
    paper = []
    for i in tqdm([*range(1000)]):
        paper.append(runSC(chromosome=[52, 143, 230, 183], args=args, demand=np.array(demand[0]),
                           ilt_list=ilt_list[0], rlt_list=rlt_list[0]))

    print(f"Paper: {np.mean(paper)}")

run()