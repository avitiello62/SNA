#!/usr/bin/python

import networkx as nx
import random
import math


# INDEPENDENT CASCADE MODEL
# Let S_0 is the set of seeds, i.e., nodes active at step 0,
# and more in general S_t is the set of nodes activated at step t.
# Moreover, we denote with S_<t the set of nodes activated before step t, i.e., S_<t is the union of S_i for i = 0, ..., t-1.
# The dynamics then proceeds as follows.
# At each time step t >= 1, for each node u in S_{t-1}, and each neighbor v not yet active (i.e., not in S_<t),
# with probability prob v is activated, i.e. v is inserted in S_t.
# The dynamics stops at the first time step t* such that S_t* is empty.
def cascade(graph, active):
    #active represents the set S_t in the description above
    if len(active) > 0:
        #newactive represents the set S_{t+1}
        newactive=set()
        for i in active:
            #This allows to keep track of S_<t, i.e. the set of nodes activated before time t
           graph.nodes[i]['active']=True
        for i in active:
           for j in graph[i]:
                if 'active' not in graph.nodes[j]:
                    r=random.random()
                    if r < graph[i][j]['prob']:
                        newactive.add(j)
        cascade(graph,newactive)
    return graph


# LINEAR THRESHOLD MODEL
# The dynamics then proceeds as follows.
# At each time step t >= 1, for each node v not yet activated (i.e., not in S_<t)
# if the fraction of activated neighbors (i.e., neighbors in S_<t)
# is above the threshold t(v), then v is activated, i.e., v is inserted in S_t.
# The dynamics stops at the first time step t* such that S_t* is empty.
def threshold(graph, active):
    #Thresholds are randomly assigned only at first step
    thresholds=nx.get_node_attributes(graph,'threshold')
    if len(thresholds) == 0:
        for i in graph.nodes():
            graph.nodes[i]['threshold'] = random.random()

    if len(active) > 0:
        newactive=set()
        for i in active:
            graph.nodes[i]['active'] = True
        for i in active:
            for j in graph[i]:
                if 'active' not in graph.nodes[j]:
                    #updating the number of active neighbors
                    if 'num_act' not in graph.nodes[j]:
                        graph.nodes[j]['num_act']=1
                    else:
                        graph.nodes[j]['num_act']+=1
                    # node activation
                    if graph.nodes[j]['num_act']/len(graph[j]) >= graph.nodes[j]['threshold']:
                        newactive.add(j)
        threshold(graph,newactive)
    return graph


# MAJORITY DYNAMICS
# At each time step we select a single node whose state does not corresponds to the state of the majority of the neighbors,
# and we fix it.
#
# Differently from the previous models, this dynamics allows to revise the state (i.e., to go from non-active to active and viceversa).
# We consider a single update at each time step. Indeed, multiple updates may lead the dynamics to never converge (e.g., the bipartite graph with each side having a different initial state).
# Note the different update order may lead to very different stable states (again, look at the case of the bipartite graph).
# This makes more complicate to forecast the outcome of dynamics running according to this dynamics.
def majority(graph, act, nact):
    # At previous step there may be either nodes that have been activated or nodes that have been dis-activated
    if len(act) > 0 or len(nact) > 0:
        for i in act:
            graph.nodes[i]['active']=True
        #updating the number of active neighbors
        for i in act:
            for j in graph[i]:
                if 'num_act' not in graph.nodes[j]:
                    graph.nodes[j]['num_act'] = 1
                else:
                    graph.nodes[j]['num_act'] += 1
        #updating the number of non active neighbors
        for i in nact:
            graph.nodes[i]['active']=False
        for i in nact:
            for j in graph[i]:
                if 'num_act' not in graph.nodes[j]:
                    graph.nodes[j]['num_act'] = 0
                else:
                    graph.nodes[j]['num_act'] -= 1
        change=False
        #We choose to update a node willing to become active before than a node willing to become non active
        #This will maximizes the number of active nodes at the end of the dynamics
        for i in graph.nodes():
            #checking if i has an incentive to be activated
            if ('active' not in graph.nodes[i] or not graph.nodes[i]['active']) and 'num_act' in graph.nodes[i] and graph.nodes[i]['num_act'] >= len(graph[i])/2:
                majority(graph,[i],[])
                change=True
                # the break serves to update a single node at each time step
                break
        if not change: #if no node is willing to become active, checks for nodes willing to become non active
            for i in graph.nodes():
                # checking if i has an incentive to be dis-activated
                if ('active' in graph.nodes[i] and graph.nodes[i]['active']) and ('num_act' not in graph.nodes[i] or graph.nodes[i]['num_act'] < len(graph[i])/2):
                    majority(graph,[],[i])
                    break
    return graph

# VOTER MODEL
# Above models are essentially deterministic models (except, for example, for the random selection of thresholds).
# There are instead purely probabilistic models, such as the voter model.
#
# In this model, at each time step we select a random node, and we update this node's state by coping the state of a random neighbor.
# The advantage of these models is that they allow "wrong" choice, that may be realistic, and useful to escape from bad unrealistic equilibria.
#
# Note that this dynamics does not terminates, unless either all nodes are active, or all nodes are not.
# For this reason we only run a limited number of steps.
def voter(graph, seed, num_steps):
    #Initialization
    for i in graph.nodes():
        if i in seed:
            graph.nodes[i]['active'] = True
        else:
            graph.nodes[i]['active'] = False

    #Update
    for t in range(num_steps):
        u=random.choice(list(graph.nodes()))
        v=random.choice(list(graph[u]))
        graph.nodes[u]['active'] = graph.nodes[v]['active']
    return graph

# Since the independent cascade and the linear threshold models are not deterministic,
# the marginal contribution cannot be directly computed, but it should be estimated
# by averaging over multiple run of the dynamics.
#
# As we will observe below the number of runs necessary to achieve a good estimation is very large,
# and this increases the running time of the greedy algorithm.
# On the other side, if the greedy algorithm does not work with a good estimation,
# its approximation can become much larger thn what it is stated below.
def marginal_influence(graph,seeds,v,dynamics, stubborness):
    # In order to have that the returned estimation is close to the real marginal contribution with high probability,
    # we have to choose a large number of runs. Specifically, to have that the estimated value is within the interval
    # [(1-eps)*true value, (1+eps)*true value] with probability at least 1-nu the number of runs must be the following one
    # num_repeat=math.ceil(2*graph.number_of_nodes()*math.ln(1/nu)/(eps*eps))
    # E.g., if we want that the estimated value is at most 5% away from the real value with probability at least 90%,
    # we should set eps = 0.05 and nu = 0.1, and we have num_repeat = 2*n*ln(10)*400 = 1842*n.
    # If, instead, we want that the estimated value is at most 1% away from the real value with probability at least 99%,
    # we should set eps = 0.01 and nu = 0.01, and we have num_repeat = 2*n*ln(100)*10000 = 92104*n.
    eps=0.05
    nu = 0.1
    num_repeat = math.ceil(2 * graph.number_of_nodes() * math.log(1 / nu) / (eps * eps))
    sumt=0
    for i in range(num_repeat):
        sumt+=len(nx.get_node_attributes(dynamics(graph, list(seeds+[v]), stubborness),'active'))
    return sumt/num_repeat

#The following greedy algorithms returns a set of seeds of size budget such that
#the (expected) number of active nodes at the end of the dynamics is a good approximation (namely, 1-1/e)
#of the maximum (expected) number of nodes that would be achieved by the best seed set of size budget,
#whenever the dynamics satisfies the following two properties:
#- Monotony: the (expected) number of active nodes at the end of the dynamics increases as the number of seed nodes increase;
#- Submodularity: the marginal contribution of a seed node (i.e., how much it increases the number of active nodes at the end of the dynamics)
#                 is larger when this seed node is added to a small seed set than when it is added to a large seed set.
#
# Interestingly, both the independent cascade model and the linear threshold model enjoy these properties.
# Hence, for these dynamics the following greedy algorithm returns a good approximation of the optimal seed set.
#
# The greedy algorithm simply works by adding at each time step the node with larger marginal contribution to the seed set
def greedy(graph,budget,dynamics, stubborness):
    seeds=[]
    while budget > 0:
        best=0
        for v in graph.nodes():
            if v not in seeds:
                #compute the marginal contribution of each node that is not yet in the seed set
                infl=marginal_influence(graph,seeds,v,dynamics, stubborness)
                #we save the one node with larger marginal contribution
                if infl >= best:
                    best=infl
                    bestV=v
        #we add the node with larger marginal contribution to the seed set
        seeds.append(bestV)
        budget-=1
    return seeds

'''
G = nx.Graph()
G.add_edge('A', 'B',prob=2/3)
G.add_edge('A', 'C',prob=2/3)
G.add_edge('B', 'C',prob=2/3)
G.add_edge('B', 'D',prob=2/3)
G.add_edge('D', 'E',prob=2/3)
G.add_edge('D', 'F',prob=2/3)
G.add_edge('D', 'G',prob=2/3)
G.add_edge('E', 'F',prob=2/3)
G.add_edge('F', 'G',prob=2/3)
seed = {'B'}
# INDEPENDENT CASCADE
print(list(nx.get_node_attributes(cascade(G, seed), 'active').keys()))

G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A', 'C')
G.add_edge('B', 'C')
G.add_edge('B', 'D')
G.add_edge('D', 'E')
G.add_edge('D', 'F')
G.add_edge('D', 'G')
G.add_edge('E', 'F')
G.add_edge('F', 'G')
seed = {'B'}
# LINEAR THRESHOLD
print(list(nx.get_node_attributes(threshold(G, seed), 'active').keys()))

G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A', 'C')
G.add_edge('B', 'C')
G.add_edge('B', 'D')
G.add_edge('D', 'E')
G.add_edge('D', 'F')
G.add_edge('D', 'G')
G.add_edge('E', 'F')
G.add_edge('F', 'G')
seed = {'B','F'}
# MAJORITY
active = nx.get_node_attributes(majority(G, seed, {}), 'active')
print([i for i in active.keys() if active[i]])

G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A', 'C')
G.add_edge('B', 'C')
G.add_edge('B', 'D')
G.add_edge('D', 'E')
G.add_edge('D', 'F')
G.add_edge('D', 'G')
G.add_edge('E', 'F')
G.add_edge('F', 'G')
seed = {'B'}
# VOTER
active = nx.get_node_attributes(voter(G, seed, 10), 'active')
print([i for i in active.keys() if active[i]])

G=nx.Graph()
G.add_edge('A','B',prob=0.66)
G.add_edge('A','C',prob=0.6)
G.add_edge('B','C',prob=0.75)
G.add_edge('B','D',prob=0.55)
G.add_edge('D','E',prob=0.7)
G.add_edge('D','F',prob=0.5)
G.add_edge('D','G',prob=0.45)
G.add_edge('E','F',prob=0.8)
G.add_edge('F','G',prob=0.66)
print(greedy(G,2,cascade))
'''
