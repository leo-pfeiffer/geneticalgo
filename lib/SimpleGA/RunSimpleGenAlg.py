from SimpleGA.FitnessFunctions import multimodal
from SimpleGA.SimpleGenAlg import SimpleGenAlg
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SimpleGA.FitnessFunctions import multimodal

mrs = np.arange(0.05, 1, 0.05).tolist()
space = np.arange(5, 51, 5).tolist()
simulations = 1000

for m in mrs:
    for s in space:
        for i in simulations:
            args = {"lower": -s,
                    "upper": s}
            max_gen = 50
            GA = SimpleGenAlg(args=args, m=m)
            GA.runAlgorithm(maxGen=max_gen)

            x_values = [x[0] for x in GA.search]
            y_values = [y[1] for y in GA.search]

            fitness_history = GA.fitness

            df = pd.DataFrame([[m] * max_gen, [s] * max_gen, x_values, y_values, fitness_history]).T
            df.columns = ["x", "y", "fitness"]

int(0)
