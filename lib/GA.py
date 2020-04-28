import numpy as np
from operator import attrgetter
from SC import returntscc, evaluate
import matplotlib.pyplot as plt


class GA:
    def __init__(self):
        self.n = 20  # number of chromosomes (= pop_size); fixed
        self.l = 4  # length of chromosomes (=agents in supply chain)
        self.par_pop = [CHROM(no=i) for i in range(1, self.n+1)] # parent population
        self.int_pop = []  # intermediate population
        self.pool = []  # mating pool; changes every iteration
        self.cr = 0.7  # crossover rate (probability)
        self.mr = 0.1  # mutation rate (probability)
        self.x = 0.2  # parameter for mutation
        self.no_gen = 0

        for chrom in self.par_pop:
            chrom.evaluate()

    def runAlgorithm(self, maxGen):
        x = []
        y = []
        while self.no_gen <= maxGen:
            self.no_gen += 1
            self.selection()
            self.crossover()
            self.mutation()
            self.survival()
            x.append(self.no_gen)
            y.append(self.par_pop[0].tscc)
            print(self.no_gen, self.par_pop[0].tscc)
            if self.no_gen % 10 == 0:
                print(self.par_pop[0].chromosome)
        plt.plot(x, y)
        plt.ylabel("TSCC")
        plt.xlabel("Generation number")
        plt.show()

    def selection(self):
        """Select chromosomes for mating pool"""
        fks = [1 / (1 + chrom.tscc) for chrom in self.par_pop]
        sumfk = sum(fks)
        probabilities = np.array([fk / sumfk for fk in fks]).cumsum()

        self.pool = [next(chrom for chrom, val in enumerate(probabilities) if val >= np.random.uniform(0, 1))]
        while len(self.pool) < self.n:
            r = np.random.uniform(0, 1)
            cn = next(chrom for chrom, val in enumerate(probabilities) if val >= r)
            if cn not in self.pool:
                self.pool.append(cn)

        self.pool = [self.par_pop[i] for i in self.pool]

    def crossover(self):
        u = np.random.uniform(0, 1)
        for i in range(int(np.ceil(self.n/2))):
            if u <= self.cr:
                cut = np.random.randint(1, self.l)
                cross1 = np.append(self.pool[i*2].chromosome[:cut], self.pool[i*2+1].chromosome[cut:])
                cross2 = np.append(self.pool[i*2+1].chromosome[:cut], self.pool[i*2].chromosome[cut:])
                self.int_pop = [CHROM(genes=cross1), CHROM(genes=cross2)]
            else:
                self.int_pop = self.pool[i*2:(i*2+1)]

    def mutation(self):
        for c, chrom in enumerate(self.int_pop):
            newGenes = []
            for g, gene in enumerate(chrom.chromosome):
                u = np.random.uniform(0, 1)
                if u <= self.mr:
                    newGenes.append(int(np.floor(gene * (1 - self.x) + gene * 2 * self.x * u)))
                else:
                    newGenes.append(gene)
            self.int_pop[c] = CHROM(genes=np.array(newGenes))

        int(0)  # should have new int_pop

    def survival(self):
        for chrom in self.int_pop:
            chrom.evaluate()

        self.par_pop = sorted(self.int_pop + self.par_pop, key=attrgetter('tscc'))[:self.n]

        for i, chrom in enumerate(self.par_pop):
            chrom.no = self.no_gen + i


class CHROM:

    def __init__(self, **kwargs):
        self.no = kwargs.get('no', -999)
        self.minRLT = np.array([1, 3, 5, 4])    # make this variable
        self.maxRLT = np.cumsum(self.minRLT[::-1])[::-1]
        self.lowerU = 20
        self.upperU = 60
        self.chromosome = kwargs.get('genes', self.generateChromosome())
        self.tscc = 0

    def generateChromosome(self):
        """generate initial chromosomes"""
        lower = self.lowerU * self.minRLT
        upper = self.upperU * self.maxRLT
        return np.array([np.random.randint(l, u + 1) for l, u in zip(lower, upper)])

    def evaluate(self):
        """Run the SC model and evaluate TSCC"""
        self.tscc = evaluate(self.chromosome)
