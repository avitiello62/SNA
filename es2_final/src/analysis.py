import networkx as nx
from networkx.readwrite.edgelist import read_edgelist
import matplotlib.pyplot as plt
import numpy as np
from utils.lesson4 import *
from utils.lesson1 import parallel_diam


from networkx.algorithms.cluster import clustering

# Function to plot degree distribution of a graph
def scatterplot_degree_distribution(graph):
    degree = nx.degree_histogram(graph)
    print("Max degree: ", len(degree))
    x = range(len(degree))
    y = [d / float(sum(degree)) for d in degree]
    plt.scatter(x, y, s=1, color=(1, 0, 0))
    plt.show()

def save_scatterplot_degree_distribution(graph, name):
    degree = nx.degree_histogram(graph)
    print("Max degree: ", len(degree))
    x = range(len(degree))
    y = [d / float(sum(degree)) for d in degree]
    plt.scatter(x, y, s=1, color=(1, 0, 0))
    #plt.show()
    plt.savefig(name)
    plt.clf()

def log_scatterplot_degree_distribution(graph):
    degree = nx.degree_histogram(graph)
    x = range(len(degree))
    y = [d / float(sum(degree)) for d in degree]
    x = np.array(x)+1
    y = np.array(y)+1
    x = np.log(x)
    y = np.log(y)
    plt.scatter(x, y, s=1, color=(1, 0, 0))
    plt.show()



net_file = '../net_6'

# Read graph from file
G = read_edgelist(net_file)
print("The net_6 has %d nodes and %d edges" % (G.number_of_nodes(), G.number_of_edges()))

# Clustering coefficient 0.07872
C = list(clustering(G).values())
C = sum(C) / len(C)
print("The net_6 has clustering coefficient: %f" % C)

scatterplot_degree_distribution(G)
log_scatterplot_degree_distribution(G)
d = parallel_diam(G,5)
print("The net_6 has diameter: ", d)


###############################
## net 6 is random?
###############################
RG = randomG(10000, 0.0016)
print("The random graph has %d nodes and %d edges" % (G.number_of_nodes(), G.number_of_edges()))
# Clustering coefficient 0.07872
C = list(clustering(RG).values())
C = sum(C) / len(C)
print("The random graph has clustering coefficient: %f" % C)
n_conn_comps = nx.number_connected_components(G)
print("The random graph has number of connected components of: %f" % n_conn_comps)

d = parallel_diam(G, 5)
print("The random graph has diameter: ", d)

scatterplot_degree_distribution(G)
log_scatterplot_degree_distribution(G)



experiments = {
    1: {"n":10000, "r":1, "k":6, "q":1, "n_trials":10},
    2: {"n":10000, "r":1, "k":7, "q":1, "n_trials":10},
    3: {"n":10000, "r":1, "k":7, "q":2, "n_trials":10},
    4: {"n":10000, "r":2, "k":1, "q":1, "n_trials":10},
    5: {"n":10000, "r":1, "k":7, "q":1.5, "n_trials":10},
    6: {"n":10000, "r":2, "k":2, "q":1, "n_trials":9},
    7: {"n":10000, "r":2, "k":2, "q":1.5, "n_trials":10},
    8: {"n":10000, "r":1.5, "k":6, "q":1.5, "n_trials":5},
    9: {"n":10000, "r":1.5, "k":7, "q":1, "n_trials":10},
    10: {"n":10000, "r":2, "k":2, "q":2, "n_trials":10},
    11: {"n":10000, "r":2, "k":5, "q":2, "n_trials":10},
    13: {"n":10000, "r":1.5, "k":8, "q":2, "n_trials":10},
}

for key in experiments.keys():
    print(key)
    value = experiments[key]
    n_trials = value["n_trials"]
    n=value["n"]
    r=value["r"]
    k=value["k"]
    q=value["q"]
    for i in range(n_trials):
        G = GenWS2DG(n, r, k, q)
        print("The provided graph has %d nodes and %d edges" % (G.number_of_nodes(), G.number_of_edges()))
        name = "exp{}_trial{}_n{}_r{}_k{}_q{}".format(str(key),str(i),str(n),str(r).replace(".",","),str(k).replace(".",","),str(q).replace(".",","))
        print("Name: ", name)
        save_scatterplot_degree_distribution(G, name)

