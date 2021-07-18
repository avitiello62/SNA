import numpy as np
import networkx as nx

def FJ_dynamics(G, b, s, num_iter=100, tolerance=1.0e-5):
    """
    Function to get FJ dynamics of a graph
    :param graph: Networkx graph
    :param b: dictionary representing the preferences of the voters
    :param s: dictionary representing the stubbornness of the voters
    :param num_iter: Number of iterations
    :param tolerance: Tolerance for the error
    :return: x if the algorithm converge, -1 otherwise
    """
    x = b
    for i in range(num_iter):
        x_new = {}
        for u in G.nodes():
            sum = 0
            for v in G[u]:
                sum += 1/len(G[u]) * x[v]
            x_new[u] = s[u] * b[u] + (1 - s[u]) * sum

        old_values = np.array(list(x.values()))
        new_values = np.array(list(x_new.values()))

        error = np.absolute(new_values - old_values).sum()
        x = x_new

        if error < tolerance:
            return x

    return -1

if __name__ == '__main__':
    import random
    G = nx.Graph()
    G.add_edge('A', 'D', weight=3)
    G.add_edge('D', 'B', weight=7)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('A', 'v', weight=7)
    G.add_edge('B', 'C')
    G.add_edge('D', 'E', weight=6)
    G.add_edge('F', 'G', weight=4)
    G.add_edge('G', 'H', weight=2)
    G.add_edge('A', 'F', weight=9)

    b = {}
    s = {}

    for v in G.nodes():
        b[v] = random.uniform(0, 1)
        s[v] = random.uniform(0, 1)

    if FJ_dynamics(G, b, s) == -1:
        print("Algorithm does not converge!")
    else:
        print("Ok!")
