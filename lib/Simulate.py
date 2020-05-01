import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from GenAlg import GenAlg
from SupplyChain import runSC

# Set GenAlg and SupplyChain parameters
rlt = np.array([1, 3, 5, 4])
hcs = np.array([8, 4, 2, 1])
scs = np.array([24, 12, 6, 3])
lowerU = 20
upperU = 60

args = {'rlt': rlt, 'hcs': hcs, 'scs': scs,
        'lower': lowerU, 'upper': upperU}

tscc = []
max_gen = 300
chromosomes = []

iterations = [*range(3)]

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

ax.text(max_gen-1, avg_tscc[0], text, fontsize=10, va="top", ha="right")

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
