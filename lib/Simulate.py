import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from GenAlg import GenAlg
from SupplyChain import runSC
"""
tscc = []
max_gen = 500
chromosomes = []

iterations = [*range(30)]

for i in tqdm(iterations):

    GA = GenAlg()
    GA.runAlgorithm(maxGen=max_gen)
    tscc.append(GA.tscc)
    chromosomes.append(GA.par_pop[0].chromosome)

avg_tscc = np.mean(tscc, axis=0)
sd_tscc = np.std(tscc, axis=0)

pd.DataFrame([avg_tscc, sd_tscc]).T.to_csv("Report.csv")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([*range(max_gen)], avg_tscc)[0]
ax.set_ylabel("TSCC")
ax.set_xlabel("Generation number")

best_chrom = chromosomes[np.argmin([x[-1] for x in tscc])].tolist()
text = "Best policy:\nRetailer: {}, Distributer: {},\nManufacturer: {}, Supplier: {}"
text = text.format(best_chrom[0], best_chrom[1], best_chrom[2], best_chrom[3])

ax.text(max_gen-1, avg_tscc[0], text, fontsize=10, va="top", ha="right")

plt.savefig("Report.png")
plt.show()
"""
x = 0
for i in range(1000):
    # tscc1 = runSC([183, 230, 143, 52])
    tscc1 = runSC([52, 143, 230, 183])
    print(tscc1)
    x += tscc1

print("\n", x/1000)

