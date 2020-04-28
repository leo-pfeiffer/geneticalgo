import numpy as np


class AGENT:

    def __init__(self, no, basestock, rlt, hcs, scs, T):
        self.no = no
        self.T = T
        self.basestock = basestock
        self.rlt = rlt
        self.holdingcost_rate = hcs
        self.shortagecost_rate = scs
        self.holdingcost_is = 0
        self.shortagecost_is = 0
        self.inventory = 0  # maybe this should be initiated with the basestock levels
        self.backlog = 0  # existing backlog
        self.order = 0    # order from downstream
        self.receive = [0] * int(np.floor(self.T * 1.5))  # received amount from upstream; account for schedule beyond T

    def updateInventory(self, amount):
        self.inventory += amount


class SC:

    def __init__(self, agents, T):
        self.agents = agents
        self.N = len(self.agents)
        self.T = T
        self.demand = np.random.randint(10, 21, self.T)
        self.scc = [0] * self.T
        self.tscc = 0

    def simulate(self):
        for t in range(0, self.T):
            # update the order info from
            for i, agent in enumerate(self.agents):
                # 1 scheduled receipt of material at agent
                agent.inventory += agent.receive[t]
                # 2 backlog is filled as much as possible
                if agent.backlog > 0:
                    newBacklog = max(agent.backlog - agent.inventory, 0)
                    # for all agents except retailer, schedule reception for downstream agent
                    if agent.no != 0:
                        downstream = self.agents[agent.no - 1]
                        downstream.receive[t + downstream.rlt] += agent.backlog - newBacklog
                    agent.inventory = max(agent.inventory - agent.backlog, 0)
                    agent.backlog = newBacklog  # check if this should be +=: probably not.
                    # maybe do shipping to customer as well ?
                # 3
                # agents receive demand/order quantity from immediate downstream member
                if agent.no == 0:
                    agent.order = self.demand[t]

                # 4
                # send order to upstream (basestock - inventory) -> received immediately
                orderQuantity = max(agent.basestock - agent.inventory, 0)
                if agent.no != 3:
                    upstream = self.agents[agent.no + 1]
                    upstream.order += orderQuantity
                    # maybe add external source
                else:
                    # supplier receives order from infinite source after rlt
                    agent.receive[t + agent.rlt] += orderQuantity

                # 5
                # order fulfillment. demand/orders of t is fulfilled based on inventory
                # if not all can be fulfilled => backlog
                shippment = min(agent.order, agent.inventory)
                agent.backlog += agent.order - shippment
                if agent.no != 0:
                    downstream = self.agents[agent.no - 1]
                    downstream.receive[t + downstream.rlt] += shippment
                    # customer receives shipping immediately

                # 6
                agent.holdingcost_is = agent.holdingcost_rate * agent.inventory
                agent.shortagecost_is = agent.shortagecost_rate * agent.backlog
            # 7
            self.scc[t] = sum([x.holdingcost_is + x.shortagecost_is for x in self.agents])

        self.tscc = np.array(self.scc).cumsum()[-1]


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
    for i, chrom in enumerate(chromosome):
        agents.append(AGENT(no=i, basestock=chrom, rlt=rlt[i], hcs=hcs[i], scs=scs[i], T=T))

    S = SC(agents=agents, T=T)
    S.simulate()

    return S.tscc