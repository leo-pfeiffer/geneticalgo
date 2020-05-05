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
    'opt': [52, 143, 230, 183]
}

a2 = {
    'hcs': np.array([8, 6, 4, 2]),
    'scs': np.array([32, 21, 12, 4]),
    'rlt': np.array([1, 3, 5, 4]),
    'opt': [54, 144, 229, 179]
}

a3 = {
    'hcs': np.array([5, 4, 3, 1]),
    'scs': np.array([10, 8, 6, 2]),
    'rlt': np.array([1, 3, 5, 4]),
    'opt': [50, 138, 223, 180]
}

b1 = {
    'hcs': np.array([8, 4, 2, 1]),
    'scs': np.array([24, 12, 6, 3]),
    'rlt': np.array([2, 3, 6, 4]),
    'opt': [95, 142, 273, 182]
}

b2 = {
    'hcs': np.array([8, 6, 4, 2]),
    'scs': np.array([32, 21, 12, 4]),
    'rlt': np.array([2, 3, 6, 4]),
    'opt': [98, 143, 272, 179]
}

b3 = {
    'hcs': np.array([5, 4, 3, 1]),
    'scs': np.array([10, 8, 6, 2]),
    'rlt': np.array([2, 3, 6, 4]),
    'opt': [91, 138, 265, 180]
}

"""
# Choose the desired setting
args = a3

tscc = []
max_gen = 500
chromosomes = []
iterations = [*range(30)]

for i in tqdm(iterations):
    GA = GenAlg(args=args)
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


# Run SC only with given base-stock

# Setup random sampling:

n_it = 10000

# w/ antithetic sampling
sample = [np.random.randint(20, 61, 1400).tolist() for i in range(int(np.floor(n_it/2)))]
anti_sample = [[80-u for u in demand] for demand in sample]
demand = sample + anti_sample

# w/o antithetic sampling
# demand = [np.random.randint(20, 61, 1400).tolist() for i in range(n_it)]

results = []
tscc = []

iterations = [*range(n_it)]

for a, arg in enumerate([a1]):
    for j in tqdm(iterations):
        tscc1 = runSC(arg['opt'], args=arg, demand=demand[j])
        tscc.append(tscc1)

    # print("\n", a, np.mean(tscc), np.std(tscc))
    results.append([np.mean(tscc), np.std(tscc)])
    tscc = []

df = pd.DataFrame(results, columns=["Mean", "SD"]).astype(int)
# print("\n", df)

ought = [[425340, 13316]]
"""
         [634072, 21369],
         [364930, 12194],
         [493501, 16069],
         [719423, 24660],
         [406393, 14161]]"""


ough_df = pd.DataFrame(ought, columns=["Mean", "SD"]).astype(int)

delta_perc = np.round((df/pd.DataFrame(ought, columns=["Mean", "SD"]) - 1), 4)

delta_abs = (df - pd.DataFrame(ought, columns=["Mean", "SD"])).astype(int)

print("\n", df, "\n", delta_perc, "\n", delta_abs)

int(0)
