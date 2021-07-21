import networkx as nx
import random
# import matplotlib.pyplot as plt
# import time


# G : Original Graph
# size : size of the sampled graph
class ForestFire():
    def __init__(self):
        self.G1 = nx.Graph()

    def forestfire(self, G, size):
        list_nodes = list(G.nodes())
        # print(len(G))
        dictt = set()
        random_node = random.sample(set(list_nodes), 1)[0]
        # print(random_node)
        q = set()   # q = set contains the distinct values
        q.add(random_node)
        while(len(self.G1.nodes()) < size):
            if(len(q) > 0):
                initial_node = q.pop()
                if(initial_node not in dictt):
                    # print(initial_node)
                    dictt.add(initial_node)
                    neighbours = list(G.neighbors(initial_node))
                    # print(list(G.neighbors(initial_node)))
                    np = random.randint(1, len(neighbours))
                    # print(np)
                    # print(neighbours[:np])
                    for x in neighbours[:np]:
                        if(len(self.G1.nodes()) < size):
                            self.G1.add_edge(initial_node, x)
                            q.add(x)
                        else:
                            break
                else:
                    continue
            else:
                random_node = random.sample(set(list_nodes) and dictt, 1)[0]
                q.add(random_node)
        q.clear()
        return self.G1


def custom_sampling_graph(graph, nodes_percentage):
    
    '''
    This function takes in input a graph and the nodes percentage and returns a sampled version of the graph only on the largest connected component (deleting nodes with degree <= 2)
    The Forest Fire Algorithm has been used to do sampling. 
    '''
    print("n. of original nodes and edges: " + str(len(graph.nodes())) + " \\ " + str(len(graph.edges())))
    
    print("SAMPLING ON THE LARGEST CONNECTED COMPONENT")
    to_be_removed = [x for  x in graph.nodes() if graph.degree(x)  <= 2]

    for x in to_be_removed :
            graph.remove_node(x)

    Gcc = max(nx.connected_components(graph), key=len)
    new_graph = graph.subgraph(list(Gcc))
    print(len(new_graph.nodes()))
    print(nx.is_connected(new_graph))
    ff = ForestFire()
    sampler = ff.forestfire(new_graph, nodes_percentage*len(new_graph.nodes()))
    
    return sampler


if __name__ == "__main__":
    graph = G = nx.complete_graph(1000)
    sampled = custom_sampling_graph(graph, 0.5)
    print("nodes of the sampled graph",len(sampled.nodes()))

