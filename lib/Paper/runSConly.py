import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.GenAlg_paper import GenAlg
from model.SupplyChain_paper import runSC
from model.SCsettings_paper import a1, a2, a3, b1, b2, b3, randomArgs, demandSample, Output
import time

paper = []
mail = []
for i in tqdm([*range(1000)]):
    paper.append(runSC(chromosome=[52, 143, 230, 183], args=a1))
    mail.append(runSC(chromosome=[52, 134, 229, 173], args=a1))

print(f"Paper: {np.mean(paper)}")
print(f"Mail: {np.mean(mail)}")