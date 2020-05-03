import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from GenAlg import GenAlg
from SupplyChain import runSC

# Default parameters from the paper

a1 = {
    'hcs': np.array([8, 4, 2, 1]),
    'scs': np.array([24, 12, 6, 3]),
    'rlt': np.array([1, 3, 5, 4]),
    'name': 'A1'
}

a2 = {
    'hcs': np.array([8, 6, 4, 2]),
    'scs': np.array([32, 21, 12, 4]),
    'rlt': np.array([1, 3, 5, 4]),
    'name': 'A2'
}

a3 = {
    'hcs': np.array([5, 4, 3, 1]),
    'scs': np.array([10, 8, 6, 2]),
    'rlt': np.array([1, 3, 5, 4]),
    'name': 'A3'
}

b1 = {
    'hcs': np.array([8, 4, 2, 1]),
    'scs': np.array([24, 12, 6, 3]),
    'rlt': np.array([2, 3, 6, 4]),
    'name': 'B1'
}

b2 = {
    'hcs': np.array([8, 6, 4, 2]),
    'scs': np.array([32, 21, 12, 4]),
    'rlt': np.array([2, 3, 6, 4]),
    'name': 'B2'
}

b3 = {
    'hcs': np.array([5, 4, 3, 1]),
    'scs': np.array([10, 8, 6, 2]),
    'rlt': np.array([2, 3, 6, 4]),
    'name': 'B3'
}

# Choose the desired setting

settings = [a1, a2, a3, b1, b2, b3]
tscc = []
max_gen = 500
chromosomes = []
iterations = [*range(30)]

df = pd.DataFrame(index=[*range(max_gen)])
plots = []

for setting in settings:

    args = setting

    for i in tqdm(iterations):
        GA = GenAlg(args=args)
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)

    avg_tscc = np.mean(tscc, axis=0)

    df = pd.concat([df, pd.DataFrame(avg_tscc, columns=[setting['name']])], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([*range(max_gen)], avg_tscc)[0]
    ax.set_ylabel("TSCC")
    ax.set_xlabel("Generation number")

    best_chrom = chromosomes[np.argmin([x[-1] for x in tscc])].tolist()
    text = "{} - Best policy:\nRetailer: {}, Distributer: {},\nManufacturer: {}, Supplier: {},\nTSCC: {}"
    text = text.format(setting['name'], best_chrom[0], best_chrom[1], best_chrom[2], best_chrom[3], avg_tscc[-1])

    ax.text(max_gen - 1, avg_tscc[0], text, fontsize=10, va="top", ha="right")

    filename_png = "Plot_{}.png".format(setting['name'])
    plt.savefig(filename_png)
    plt.show()

df.index = [*range(1, max_gen+1)]
df.to_csv("TSCC_all_settings")

# Create plots per category

for category in ["A", "B"]:
    if category == "A":
        ax = df.iloc[:, 0:3].plot()
    else:
        ax = df.iloc[:, 3:].plot()

    ax.set_ylabel("TSCC")
    ax.set_xlabel("Generation number")

    filename_png = "Plot_{}.png".format(category)
    plt.savefig(filename_png)
    plt.show()
