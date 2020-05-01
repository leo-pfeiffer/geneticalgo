import numpy as np


class Agent:

    def __init__(self, no, basestock, rlt, hcs, scs):
        self.no = no
        self.T = 1200
        self.basestock = basestock
        self.rlt = rlt
        self.holdingcost_rate = hcs
        self.shortagecost_rate = scs
        self.holdingcost_is = 0
        self.shortagecost_is = 0
        self.onHandInventory = basestock
        self.onOrderInventory = 0
        self.backlog = 0
        self.order = 0  # order from downstream
        self.receive = [0] * int(np.floor(self.T * 1.5))  # received from upstream; account for schedule beyond T


class SupplyChain:

    def __init__(self, agents, args):
        self.agents = agents
        self.N = len(self.agents)
        self.T = 1200
        self.demand = np.random.randint(args['lower'], args['upper'] + 1, self.T)
        self.scc = [0] * self.T
        self.tscc = 0

    def simulate(self):
        received = 0
        ordered = 0
        for t in range(0, self.T):

            # update the order info from
            for i, agent in enumerate(self.agents):

                # define upstream and downstream
                if agent.no != 0:
                    downstream = self.agents[agent.no - 1]
                if agent.no != 3:
                    upstream = self.agents[agent.no + 1]

                # 1 scheduled receipt of material at agent
                agent.onHandInventory += agent.receive[t]
                agent.onOrderInventory -= agent.receive[t]

                # 2 backlog is filled as much as possible
                if agent.backlog > 0:
                    newBacklog = max(agent.backlog - agent.onHandInventory, 0)

                    # for all agents except retailer, schedule reception for downstream agent and do shipment
                    if agent.no != 0:
                        downstream.receive[t + downstream.rlt] += agent.backlog - newBacklog

                    # update inventory information
                    agent.onHandInventory = max(agent.onHandInventory - agent.backlog, 0)
                    agent.backlog = newBacklog

                # 3 agents receive demand/order quantity from immediate downstream member
                if agent.no == 0:
                    agent.order = self.demand[t]

                # 4
                # send order to upstream -> received immediately
                # calculate end of period onHandInventory
                shipment = min(agent.order, agent.onHandInventory)
                agent.onHandInventory -= shipment

                orderQuantity = max(agent.basestock + agent.backlog - agent.onHandInventory - agent.onOrderInventory, 0)

                if agent.no != 3:
                    upstream.order = orderQuantity
                else:
                    # supplier receives order from infinite source after rlt
                    agent.receive[t + agent.rlt] += orderQuantity

                # 5
                # order fulfillment. demand/order of t is fulfilled based on inventory
                # if not all can be fulfilled => backlog
                agent.backlog += agent.order - shipment

                if agent.no != 0:
                    downstream.receive[t + downstream.rlt] += shipment
                    # customer receives shipment immediately

                # 6 update inventory and calculate local costs
                agent.onOrderInventory += orderQuantity
                agent.holdingcost_is = agent.holdingcost_rate * agent.onHandInventory
                agent.shortagecost_is = agent.shortagecost_rate * agent.backlog

                if agent.no in [0]:
                    received += agent.receive[t]
                    ordered += orderQuantity

            # 7 calculate supply chain contexts
            self.scc[t] = sum([x.holdingcost_is + x.shortagecost_is for x in self.agents])

        self.tscc = np.array(self.scc).cumsum()[-1]


def returnTSCC(chromosome):
    goal = [12, 13, 4, 17]
    return sum([abs(x - y) for x, y in zip(chromosome, goal)])


def runSC(chromosome, args):
    # Initialise
    agents = []
    rlt = args['rlt']
    hcs = args['hcs']
    scs = args['scs']
    for i, chrom in enumerate(chromosome):
        agents.append(Agent(no=i, basestock=chrom, rlt=rlt[i], hcs=hcs[i], scs=scs[i]))

    S = SupplyChain(agents=agents, args=args)
    S.simulate()

    return S.tscc
