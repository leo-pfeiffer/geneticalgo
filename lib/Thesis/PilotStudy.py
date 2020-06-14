import concurrent.futures
import numpy as np
import pandas as pd
from model.GenAlg_thesis import GenAlg
from model.SCsettings_thesis import s1, s2, s3, s4, randomArgs, demandSample
from tqdm import tqdm
import time
import datetime
from twilio.rest import Client

client = Client("AC5bd00437e693681bdc1e4ba2beb424aa", "f7eaefa9b807b3304b02cc2b5c319ecb")
client.messages.create(to="+4917645813979",
                       from_="+17024257635",
                       body="Simulation started.")
n_it = 20
T = 1200
lower = 20
upper = 60
tasks = 12
max_gen = 25
chromosomes = []

demand = demandSample(T, lower, upper, n_it, antithetic=True)

results = pd.DataFrame(columns=["MX", "MP", "CR", "RX", "TSCC"])

iterations = [*range(n_it)]

# mxs = [0.1, 0.2, 0.3, 0.4, 0.5]
# mps = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
# crs = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

# mxs = [0.1, 0.2, 0.3]
# mps = [0.6, 0.7, 0.8]
# crs = [0.7, 0.8, 0.9]
rxs = [0.05, 0.1, 0.15, 0.2]

mxs = [0.2]
mps = [0.7]
crs = [0.8]


def ga_process(its, demand, arg, mx, mp, cr, rx):
    chromosomes = []
    tscc = []
    for i in tqdm([*range(its)]):
        GA = GenAlg(args=arg, demand=demand[i], mx=mx, mp=mp, cr=cr, rechenberg=True, rx=rx)
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        chromosomes.append(GA.par_pop[0].chromosome)
    return chromosomes, tscc


def ga_process_wrapper(args):
    return ga_process(*args)


t0 = time.time()

# arg = s1

geneticalgorithm = True

for arg in [s1, s2, s3, s4]:
    for rx in rxs:
        for mx in mxs:
            for mp in mps:
                for cr in crs:
                    start = time.time()

                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        demands = [a.tolist() for a in np.array_split(np.array(demand), tasks)]
                        args = ((len(d), d, arg, mx, mp, cr, rx) for d in demands)
                        return_split = executor.map(ga_process_wrapper, args)
                        tscc = []
                        chromosomes = []
                        for ret in return_split:
                            chromosomes += ret[0]
                            tscc += ret[1]

                    avg_tscc = np.mean(tscc, axis=0)

                    tscc = avg_tscc[-1]
                    elapsed = time.time() - start

                    row = {"MX": mx, "MP": mp, "CR": cr, "RX": rx, "TSCC": tscc}
                    print(elapsed, "\n", row, "\n")
                    print(datetime.datetime.now().strftime("%H:%M:%S"))
                    results = results.append(row, ignore_index=True)

                client.messages.create(to="+4917645813979",
                                       from_="+17024257635",
                                       body="Last done: MX: {}, MP: {}, CR: {}, RX: {}, TSCC: {}".format(mx, mp, cr, rx, tscc))

    if arg == s1:
        filename = "Rechenberg_S1.csv"
    elif arg == s2:
        filename = "Rechenberg_S2.csv"
    elif arg == s3:
        filename = "Rechenberg_S3.csv"
    elif arg == s4:
        filename = "Rechenberg_S4.csv"
    results.to_csv(filename, header=True, index=True)
    print("-----------Next iteration-------------")
    elapsed = (time.time() - t0) / 60

    client.messages.create(to="+4917645813979",
                           from_="+17024257635",
                           body="Simulation done.\nElapsed: {}\n{} saved.".format(elapsed, filename))
