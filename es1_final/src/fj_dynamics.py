import numpy as np

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