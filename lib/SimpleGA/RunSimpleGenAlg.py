from SimpleGA.FitnessFunctions import multimodal
from SimpleGA.SimpleGenAlg import SimpleGenAlg
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SimpleGA.FitnessFunctions import multimodal

args = {"lower": -30,
        "upper": 30}

max_gen = 200
GA = SimpleGenAlg(args=args)
GA.runAlgorithm(maxGen=max_gen)

x_values = [x[0] for x in GA.search]
y_values = [y[1] for y in GA.search]

chrom = GA.search[-1]
fitness = GA.fitness[-1]
fitness_history = GA.fitness

# plt.scatter(x_values, y_values)

df = pd.DataFrame([x_values, y_values, fitness_history]).T
df.columns = ["x", "y", "fitness"]
df.plot()
plt.show()

int(0)