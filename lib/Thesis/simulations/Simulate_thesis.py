import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.GenAlg_thesis import GenAlg
from model.SupplyChain_thesis import runSC
from model.SCsettings_thesis import a1, a2, a3, b1, b2, b3, randomArgs, demandSample, Output
import time

"""
# Choose the desired setting
args = a1

n_it = 30
T = 1200
lower = 20
upper = 60

demand = demandSample(T, lower, upper, n_it, antithetic=True)

tscc = []
max_gen = 500
chromosomes = []
iterations = [*range(n_it)]

for i in tqdm(iterations):
    GA = GenAlg(args=args, demand=demand[i])
    GA.runAlgorithm(maxGen=max_gen)
    tscc.append(GA.tscc)
    chromosomes.append(GA.par_pop[0].chromosome)

avg_tscc = np.mean(tscc, axis=0)
sd_tscc = np.std(tscc, axis=0)

# pd.DataFrame([avg_tscc, sd_tscc]).T.to_csv("Report.csv")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([*range(max_gen)], avg_tscc)[0]
ax.set_ylabel("TSCC")
ax.set_xlabel("Generation number")

best_chrom = chromosomes[np.argmin([x[-1] for x in tscc])].tolist()
text = "Best policy:\nRetailer: {}, Distributer: {},\nManufacturer: {}, Supplier: {}"
text = text.format(best_chrom[0], best_chrom[1], best_chrom[2], best_chrom[3])

ax.text(max_gen - 1, avg_tscc[0], text, fontsize=10, va="top", ha="right")

# plt.savefig("Report.png")
plt.show()


"""
"""
# Run SC only with given base-stock

# Setup random sampling:

n_it = 100
T = 1200
lower = 20
upper = 60

demand = demandSample(T, lower, upper, n_it, antithetic=True)

results = []
tscc = []
ought = []

iterations = [*range(n_it)]

start = time.time()

for a, arg in enumerate([a1]):
    for j in tqdm(iterations):
        # arg = randomArgs()
        tscc1 = runSC(arg['opt'], args=arg, demand=demand[j])
        tscc.append(tscc1)

    results.append([np.mean(tscc), np.std(tscc)])
    tscc = []
    ought.append(arg['ought'])

elapsed = time.time() - start

df, ought_df, delta_perc, delta_abs = Output(results, ought, elapsed)
"""

aa, bb, cc, dd, ee, ss = 0, 0, 0, 0, 0, 0

for i in range(1000):
    args = a1
    demand = np.random.randint(20, 61, 1200).tolist()
    tscc, a, b, c, d, e, s = runSC(args['opt'], args=args, demand=demand)
    aa += a
    bb += b
    cc += c
    dd += d
    ee += e
    ss += s

s = aa + bb + cc + dd + ee

print("A", aa, "\nB", bb, "\nC", cc, "\nD", dd, "\nE", ee, "\nOverhead", ss, "-", s)
int(0)
