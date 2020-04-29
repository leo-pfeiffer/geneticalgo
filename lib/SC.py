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
        self.onHandInventory = basestock  # maybe this should be initiated with the 0 ?
        self.onOrderInventory = 0  # keep track of orders that haven't arrived yet
        self.backlog = 0  # existing backlog
        self.order = 0  # order from downstream
        self.receive = [0] * int(np.floor(self.T * 1.5))  # received amount from upstream; account for schedule beyond T


class SC:

    def __init__(self, agents, T):
        self.agents = agents
        self.N = len(self.agents)
        self.T = T
        self.demand = np.random.randint(20, 61, self.T)
        self.scc = [0] * self.T
        self.tscc = 0

    def simulate(self):
        inv = []
        # print("onHandInventory, onOrderInventory, Order, Backlog, Receive(t)")
        received = 0
        ordered = 0
        for t in range(0, self.T):
            inv_agent = []
            # update the order info from
            for i, agent in enumerate(self.agents):
                inv_agent.append(agent.onHandInventory)
                # 1 scheduled receipt of material at agent
                agent.onHandInventory += agent.receive[t]
                agent.onOrderInventory -= agent.receive[t]
                # 2 backlog is filled as much as possible
                if agent.backlog > 0:
                    newBacklog = max(agent.backlog - agent.onHandInventory, 0)
                    # for all agents except retailer, schedule reception for downstream agent
                    if agent.no != 0:
                        downstream = self.agents[agent.no - 1]
                        downstream.receive[t + downstream.rlt] += agent.backlog - newBacklog
                    agent.onHandInventory = max(agent.onHandInventory - agent.backlog, 0)
                    agent.backlog = newBacklog

                # 3 agents receive demand/order quantity from immediate downstream member
                if agent.no == 0:
                    agent.order = self.demand[t]

                # 4
                # send order to upstream (basestock - onHandInventory) -> received immediately
                ##shipment = min(agent.order, agent.onHandInventory) #####
                ##agent.onHandInventory -= shipment #####

                orderQuantity = max(agent.basestock + agent.backlog - agent.onHandInventory - agent.onOrderInventory, 0)

                if agent.no != 3:
                    upstream = self.agents[agent.no + 1]
                    upstream.order = orderQuantity
                    # maybe add external source
                else:
                    # supplier receives order from infinite source after rlt
                    agent.receive[t + agent.rlt] += orderQuantity

                # 5
                # order fulfillment. demand/orders of t is fulfilled based on inventory
                # if not all can be fulfilled => backlog
                shipment = min(agent.order, agent.onHandInventory)
                agent.backlog += agent.order - shipment

                if agent.no != 0:
                    downstream = self.agents[agent.no - 1]
                    downstream.receive[t + downstream.rlt] += shipment
                    # customer receives shipment immediately

                # 6 update inventory and calculate local costs
                agent.onHandInventory -= shipment
                agent.onOrderInventory += orderQuantity
                agent.holdingcost_is = agent.holdingcost_rate * agent.onHandInventory
                agent.shortagecost_is = agent.shortagecost_rate * agent.backlog

                if agent.no in [0]:
                    received += agent.receive[t]
                    ordered += orderQuantity
                    print(t, agent.no, "R_total:", received, "O_total:", ordered, "R_cur:", agent.receive[t], "O_cur:", orderQuantity,
                          "Demand:", self.demand[t], "onHandInventory:", agent.onHandInventory, "onOrderInventory:", agent.onOrderInventory)

                    # Problem: We receive more than we order so we get negative onOrderInventory after some time

            # 7 calculate supply chain contexts
            self.scc[t] = sum([x.holdingcost_is + x.shortagecost_is for x in self.agents])
            inv.append(inv_agent)

            # print(t, [x.onHandInventory for x in self.agents], [x.onOrderInventory for x in self.agents],
            #      [x.order for x in self.agents], [x.backlog for x in self.agents], [x.receive[t] for x in self.agents])

        self.tscc = np.array(self.scc).cumsum()[-1]


def returntscc(chromosome):
    goal = [12, 13, 4, 17]
    return sum([abs(x - y) for x, y in zip(chromosome, goal)])


def evaluate(chromosome):
    # Initialise
    agents = []
    T = 1200
    rlt = np.array([1, 3, 5, 4])
    hcs = np.array([8, 4, 2, 1])
    scs = np.array([24, 12, 6, 3])
    for i, chrom in enumerate(chromosome):
        agents.append(AGENT(no=i, basestock=chrom, rlt=rlt[i], hcs=hcs[i], scs=scs[i], T=T))

    S = SC(agents=agents, T=T)
    S.simulate()

    return S.tscc
