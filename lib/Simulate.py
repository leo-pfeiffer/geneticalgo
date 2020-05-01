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
    'rlt': np.array([1, 3, 5, 4])
}

a2 = {
    'hcs': np.array([8, 6, 4, 2]),
    'scs': np.array([32, 21, 12, 4]),
    'rlt': np.array([1, 3, 5, 4])
}

a3 = {
    'hcs': np.array([10, 8, 6, 2]),
    'scs': np.array([24, 12, 6, 3]),
    'rlt': np.array([1, 3, 5, 4])
}

b1 = {
    'hcs': np.array([8, 4, 2, 1]),
    'scs': np.array([24, 12, 6, 1]),
    'rlt': np.array([2, 3, 6, 4])
}

b2 = {
    'hcs': np.array([8, 6, 4, 2]),
    'scs': np.array([32, 21, 12, 4]),
    'rlt': np.array([2, 3, 6, 4])
}

b3 = {
    'hcs': np.array([5, 4, 3, 1]),
    'scs': np.array([10, 8, 6, 2]),
    'rlt': np.array([2, 3, 6, 4])
}

# Choose the desired setting
args = a1

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
x = 0
for i in range(1000):
    tscc1 = runSC([52, 143, 230, 183], args=args)
    print(tscc1)
    x += tscc1

print("\n", x/1000)
"""
