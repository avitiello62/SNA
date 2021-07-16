import networkx as nx
import itertools as it
import os
import sys
import numpy as np


def FJ_dynamics(graph, b, s, num_iter=100, tolerance=1.0e-5):
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
        for u in graph.nodes():
            sum = 0
            for v in graph[u]:
                sum += 1 / len(graph[u]) * x[v]
            x_new[u] = s[u] * b[u] + (1 - s[u]) * sum

        old_values = np.array(list(x.values()))
        new_values = np.array(list(x_new.values()))

        error = np.absolute(new_values - old_values).sum()
        x = x_new

        if error < tolerance:
            return x

    return -1


def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


def custom_BFS_full(graph, u):
    level = 1
    n = graph.number_of_nodes()
    clevel = [u]
    visited = []
    visited.append(u)
    dist = {}

    while len(visited) < n:
        nlevel = []
        if len(clevel) == 0 and level == 100:
            sys.exit()
        while len(clevel) > 0:
            c = clevel.pop()
            for v in graph[c]:
                if v not in visited:
                    visited.append(v)
                    nlevel.append(v)
                    dist[v] = level
        level += 1
        clevel = nlevel

    return list(dist.keys()), list(dist.values())


def f_func(dist):
    return 1 / (1 + dist)


def shapley_closeness(graph):
    """

    :param graph:
    :param C: A coalition of players, it should be iterable
    :return:
    """
    shapley = {}

    for v in graph.nodes():
        shapley[v] = 0

    test = 0
    for v in graph.nodes():

        test += 1
        nodes, distances = custom_BFS_full(graph, v)
        index = len(nodes) - 1
        sum = 0
        prevDistance = -1
        prevSV = -1

        while index > 0:
            if distances[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = (f_func(distances[index]) / (1 + index)) - sum

            shapley[nodes[index]] += currSV
            sum += f_func(distances[index]) / (index * (1 + index))
            prevDistance = distances[index]
            prevSV = currSV
            index -= 1
        shapley[v] += f_func(0) - sum

    return shapley


def get_random_graph(nodes, edges, connected=True):
    graph = nx.gnm_random_graph(nodes, edges, seed=11, directed=False)

    if connected:
        remove_small_components(graph)

    # Makes so that the nodes are strings and not ints.
    # Strings are iterable and can be easily converted into sets
    # This operation takes a little bit of time, but they do not impact any algorithm
    # because it happens before the application of any algorithm
    # This will also speed up the algorithms which require sets and frozensets, so to avoid the
    # conversion from int to string during the execution of the algorithms
    mapping = {}
    for node in graph.nodes():
        mapping[node] = str(node)
    nx.relabel_nodes(graph, mapping, False)  # Re-labelling is done in-place
    return graph


def remove_small_components(graph):
    max = 0
    for component in list(nx.connected_components(graph)):
        if max < len(component):
            max = len(component)

    for component in list(nx.connected_components(graph)):
        if len(component) < max:
            for node in component:
                graph.remove_node(node)


def get_biggest_subgraph(graph):
    max = 0
    for component in list(nx.connected_components(graph)):
        if max < len(component):
            max = len(component)

    for component in list(nx.connected_components(graph)):
        if len(component) < max:
            for node in component:
                graph.remove_node(node)
    return graph
