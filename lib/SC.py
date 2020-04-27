import numpy as np


class AGENT:

    def __init__(self, no, basestock, rlt, hcs, scs, T):
        self.no = no
        self.T = T
        self.basestock = basestock
        self.rlt = rlt
        self.holdingcost = hcs
        self.shortagecost = scs
        self.inventory = 0
        self.order = [0]*self.T    # order from downstream
        self.receive = [0]*self.T  # received amount from upstream
        self.backlog = [0]*self.T  # existing backlog

    def updateInventory(self, amount):
        self.inventory += amount


class SC:

    def __init__(self, agents, T):
        self.agents = agents
        self.N = len(self.agents)
        self.T = T
        self.demand = np.random.randint(10, 21, self.T)
        self.tscc = 0

    def simulate(self):
        for t in range(0, self.T):
            # update the order info from
            self.agents[0].order[0] += self.demand[0]
            for i, agent in enumerate(self.agents):
                # 1
                agent.inventory += agent.receive[t]
                # 2
                if agent.no == 0:
                    pass
                    # if backlog & inventory >= backlog: ship to customer
                    # update inventory
                else:
                    pass
                    # if backlog & inventory >= backlog: ship to i-1
                    # update inventory, update [i-1].receive at t  + [i-1].rlt
                # 3
                # agents receive demand/order quantity from immediate downstream member
                # 4
                # send order to upstream (basestock - inventory) -> received immediately
                # 5
                # order fulfillment. demand/orders of t is fulfilled based on inventory
                # if not all can be fulfilled => backlog
                # downstream member receives shippment after delay of [i-1].rtl unless customer (immediate)
                # 6
                # update backlog & inventory. Calculate shortage & holding cost for agent i
            # 7
            # calculate TSCC



def returntscc(chromosome):
    goal = [12, 13, 4, 17]
    return sum([abs(x - y) for x, y in zip(chromosome, goal)])


def evaluate(chromosome):
    # Initialise
    agents = []
    T = 1200
    rlt = np.array([1, 2, 4, 8])
    hcs = np.array([1, 2, 4, 8])
    scs = np.array([3, 6, 12, 24])
    for i, chrom in chromosome:
        agents.append(AGENT(no=i, basestock=chrom[i], rlt=rlt[i], hcs=hcs[i], scs=scs[i], T=T))

    S = SC(agents=agents, T=T)
    S.simulate()

    return S.tscc
