import numpy as np
from operator import attrgetter
from tqdm import tqdm
from random import uniform, shuffle
from copy import copy

from SimpleGA.FitnessFunctions import multimodal


class SimpleGenAlg:
    def __init__(self, args):
        self.args = args
        self.n = 4  # number of chromosomes (= pop_size); fixed
        self.l = 2  # length of chromosomes (=agents in supply chain)
        self.par_pop = [Chrom(no=i, args=args, l=self.l) for i in range(1, self.n + 1)]  # parent population
        self.int_pop = []  # intermediate population
        self.pool = []  # mating pool; changes every iteration
        self.cr = 0.7  # crossover rate (probability)
        self.mr = 0.1  # mutation rate (probability)
        self.x = 0.2  # parameter for mutation
        self.no_gen = 0
        self.fitness = []  # save fitness of each generation for analysis
        self.search = []    # hold history of best solution

        for chrom in self.par_pop:
            chrom.evaluate()

    def runAlgorithm(self, maxGen):
        pbar = tqdm(maxGen)
        while self.no_gen < maxGen:
            self.no_gen += 1
            # self.selection()
            self.crossover()
            self.mutation()
            self.survival()

            self.fitness.append(self.par_pop[0].fitness)  # save fitness of current iteration
            self.search.append(self.par_pop[0].chromosome)  # save best chromosome
            pbar.update(1)

    """
    def selection(self):
        # Select chromosomes for mating pool -> crossover
        fks = [1 / (1 + chrom.fitness) for chrom in self.par_pop]
        sumfk = sum(fks)
        probabilities = np.array([fk / sumfk for fk in fks]).cumsum()

        self.pool = [next(chrom for chrom, val in enumerate(probabilities) if val >= np.random.uniform(0, 1))]
        while len(self.pool) < self.n:
            r = np.random.uniform(0, 1)
            cn = next(chrom for chrom, val in enumerate(probabilities) if val >= r)
            if cn not in self.pool:
                self.pool.append(cn)

        self.pool = [self.par_pop[i] for i in self.pool]
    """

    def crossover(self):
        self.pool = copy(self.par_pop)
        shuffle(self.pool)
        for i in range(int(np.ceil(self.n/2))):
            cut = np.random.randint(1, self.l)
            cross1 = np.append(self.pool[i*2].chromosome[:cut], self.pool[i*2+1].chromosome[cut:])
            cross2 = np.append(self.pool[i*2+1].chromosome[:cut], self.pool[i*2].chromosome[cut:])
            self.int_pop += [Chrom(genes=cross1, args=self.args, l=self.l), Chrom(genes=cross2, args=self.args, l=self.l)]

    def mutation(self):
        for c, chrom in enumerate(self.int_pop):
            newGenes = []
            for g, gene in enumerate(chrom.chromosome):
                u = np.random.uniform(0, 1)
                if u <= self.mr:
                    newGenes.append(gene * (1 - self.x) + gene * 2 * self.x * u)
                else:
                    newGenes.append(gene)
            self.int_pop[c] = Chrom(genes=np.array(newGenes), args=self.args, l=self.l)
            self.int_pop[c].evaluate()

    def survival(self):
        """Roulette wheel selection"""
        pool = self.int_pop + self.par_pop
        fks = [1 / (1 + chrom.fitness) for chrom in pool]
        sumfk = sum(fks)
        probabilities = np.array([fk / sumfk for fk in fks]).cumsum()

        new_par = [next(chrom for chrom, val in enumerate(probabilities) if val >= np.random.uniform(0, 1))]
        while len(new_par) < self.n:
            r = np.random.uniform(0, 1)
            cn = next(chrom for chrom, val in enumerate(probabilities) if val >= r)
            if cn not in new_par:
                new_par.append(cn)

        self.par_pop = [pool[i] for i in new_par]

        for i, chrom in enumerate(self.par_pop):
            chrom.no = self.no_gen + i


class Chrom:

    def __init__(self, **kwargs):
        self.args = kwargs.get('args')
        self.no = kwargs.get('no', -999)
        self.l = kwargs.get('l', 2)
        self.chromosome = kwargs.get('genes', self.generateChromosome())
        self.fitness = np.inf

    def generateChromosome(self):
        """generate initial chromosomes"""
        return np.array([uniform(self.args['lower'], self.args['upper']) for i in range(self.l)])

    def evaluate(self):
        """Call fitness function"""
        self.fitness = multimodal(self.chromosome)
