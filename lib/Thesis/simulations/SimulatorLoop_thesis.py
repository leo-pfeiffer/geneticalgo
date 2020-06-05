import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.GenAlg_thesis import GenAlg
from model.SupplyChain_thesis import runSC
from model.SCsettings_thesis import a1, a2, a3, b1, b2, b3, randomArgs, demandSample
import time

# Choose the desired setting

settings = [a1, a2, a3, b1, b2, b3]
tscc = []
max_gen = 500
chromosomes = []
iterations = [*range(30)]

df = pd.DataFrame(index=[*range(max_gen)])
plots = []

start = time.time()

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

elapsed = time.time() - start

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
