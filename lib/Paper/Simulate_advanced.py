import numpy as np
import pandas as pd
from tqdm import tqdm
from model.GenAlg_advanced import GenAlg
from model.SCsettings_paper import a1, a2, a3, b1, b2, b3, advanced_A, advanced_B, randomArgs, demandSample, Output


def run2(args, name):
    """This allows for nonzero lead times"""
    tscc = []
    max_gen = 300
    chromosomes = []
    n_it = 30
    iterations = [*range(n_it)]
    demand = demandSample(1200, 20, 60, n_it, antithetic=True)

    for i in tqdm(iterations):
        GA = GenAlg(args=args, demand=demand)
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)

    avg_tscc = np.mean(tscc, axis=0)
    sd_tscc = np.std(tscc, axis=0)

    pd.DataFrame([avg_tscc, sd_tscc]).T.to_csv(f"Report_{name}.csv")
    a = [x[-1] for x in tscc]
    b = [list(x) for x in chromosomes]
    c = [[a[i]] + b[i] for i in range(len(a))]
    pd.DataFrame(c).to_csv(f"Chromosomes_{name}.csv")


# run(a1, 'a1')
# run(a3, 'a3')
# run(b2, 'b2')

run2(advanced_A, "advanded_A")
run2(advanced_B, "advanded_B")




"""

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([*range(max_gen)], avg_tscc)[0]
ax.set_ylabel("TSCC")
ax.set_xlabel("Generation number")

best_chrom = chromosomes[np.argmin([x[-1] for x in tscc])].tolist()
text = "Best policy:\nRetailer: {}, Distributer: {},\nManufacturer: {}, Supplier: {}\nTSCC: {}"
text = text.format(best_chrom[0], best_chrom[1], best_chrom[2], best_chrom[3], avg_tscc[-1])

ax.text(max_gen - 1, avg_tscc[0], text, fontsize=10, va="top", ha="right")

# plt.savefig("Report.png")
plt.show()

"""
"""

# Run SC only with given base-stock

# Setup random sampling:

n_it = 1000
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
