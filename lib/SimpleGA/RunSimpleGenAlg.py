from SimpleGA.FitnessFunctions import multimodal
from SimpleGA.SimpleGenAlg import SimpleGenAlg
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from SimpleGA.FitnessFunctions import multimodal

# mrs = np.arange(0.05, 1, 0.05).tolist()
mrs = np.arange(0.05, 1, 0.1).tolist()
space = np.arange(5, 21, 5).tolist()
simulations = 10
delta = 0.1
df = pd.DataFrame()


def binner(x, delta):
    v1 = 3 * math.sqrt(3 / 10)
    v2 = math.sqrt(23 / 10)
    # Could outsource this
    min1 = multimodal([v1, -v1])
    min2 = multimodal([-v1, v1])
    min3 = multimodal([-v2, -v2])
    min4 = multimodal([v2, v2])
    locmax = multimodal([0, 0])

    if abs(x - min1)/min1 <= delta:
        return 1
    elif abs(x - min2)/min2 <= delta:
        return 2
    elif abs(x - min3)/min3 <= delta:
        return 3
    elif abs(x - min4)/min4 <= delta:
        return 4
    elif abs(x - locmax)/locmax <= delta:
        return 5
    else:
        return 0


def analyseSimulation(**kwargs):
    df = kwargs.get('df')
    simulations = kwargs.get('simulations')
    data = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5])
    params = []
    for m in mrs:
        for s in space:
            params.append((m, s))
            freq = df[(df.MR == m) & (df.Search == s)].Bin.value_counts()/simulations
            data = data.append(freq.to_dict(), ignore_index=True).fillna(0)
            chance = data[data[2] > 0.5]

    index = data[data[2] > 0.5].index
    for i in index:
        print(params[i])
        print(data.iloc[i])
        print("---------")

    return data


if __name__ == "__main__":
    pbar = tqdm(len(mrs))
    for m in mrs:
        for s in space:
            for i in range(simulations):
                args = {"lower": -s,
                        "upper": s}
                max_gen = 25
                GA = SimpleGenAlg(args=args, m=m)
                GA.runAlgorithm(maxGen=max_gen)
                fitness = GA.fitness[-1]
                b = binner(fitness, delta)
                x = GA.search[-1][0]
                y = GA.search[-1][1]
                data = {'MR': m, 'Search': s, 'Sim': i, 'Bin': b, 'x': x, 'y': y, 'fitness': fitness}
                df = df.append(data, ignore_index=True)
        pbar.update(1)

    # df.to_csv("SimulationResults.csv")
    data = analyseSimulation(df=df, simulations=simulations)
    int(0)


# x_values = [x[0] for x in GA.search]
# y_values = [y[1] for y in GA.search]

# fitness_history = GA.fitness
# df = pd.DataFrame([[m] * max_gen, [s] * max_gen, x_values, y_values, fitness_history]).T
# df.columns = ["x", "y", "fitness"]

