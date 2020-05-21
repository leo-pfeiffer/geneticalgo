from SimpleGA.FitnessFunctions import multimodal
from SimpleGA.SimpleGenAlg import SimpleGenAlg
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from SimpleGA.FitnessFunctions import multimodal

args = {"lower": -10,
        "upper": 10}

max_gen = 1000
GA = SimpleGenAlg(args=args)
GA.runAlgorithm(maxGen=max_gen)

x_values = [x[0] for x in GA.search]
y_values = [y[1] for y in GA.search]

chrom = GA.search[-1]
fitness = GA.fitness[-1]

plt.scatter(x_values, y_values)
plt.show()

int(0)