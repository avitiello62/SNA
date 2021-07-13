import os
import sys

import networkx as nx
from es1.src.degree import degree
from utils.priorityq import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt

def shapley_degree(G, C=None):
    """
    Shapley value for the characteristic function
        value(C) = |C| + |N(C)|,
    where N(C) is the set of nodes outside C with at least one neighbor in C

    :param G: Unweighted and undirected Networkx graph
    :param C: A coalition of players, it should be iterable
    :return: Shapley value for the characteristic function for all nodes of the coalition
    """

    if C is None:
        return 0

    deg = degree(G)

    # Shapley values
    sv = {}

    for v in C:
        sv[v] = 1 / (1 + deg[v])
        for u in G.neigbors(v):
            sv[v] += 1 / (1 + deg[u])

    return sv

def shapley_threshold(G, k, C=None):
    """
    Shapley value for the characteristic function
        value(C) = |C| + |N(C,k)|,
    where N(C,k) is the set of nodes outside C with at least k neighbors in C

    :param G: Unweighted and undirected Networkx graph
    :param C: A coalition of players, it should be iterable
    :param k: Threshold
    :return: Shapley value for the characteristic function for all nodes of the coalition
    """

    if C is None:
        return 0

    deg = degree(G)

    # Shapley values
    sv = {}

    for v in C:
        sv[v] = min(1, (k/(1+deg[v])))
        for u in G.neighbors(v):
            sv[v] += max(0, ((deg[u] - k + 1)/(deg[u] * (1 + deg[u]))))

    return sv

def shapley_closeness(G):
    """
    :param G: Weighted networkx graph
    :param f: A function for the distance
    :return: Shapley value for the characteristic function for all nodes
    """
    #Initialise
    shapley = {}

    for v in G.nodes():
        shapley[v] = 0

    for v in G.nodes():
        distances, nodes = dijkstra(v, G)
        index = len(nodes) - 1
        sum = 0
        prevDistance = -1
        prevSV = -1

        while index > 0:
            if distances[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = (f_dist(distances[index])/(1+index)) - sum

            shapley[nodes[index]] += currSV
            sum += f_dist(distances[index])/(index*(1+index))
            prevDistance = distances[index]
            prevSV = currSV
            index -= 1
        shapley[v] += f_dist(0) - sum

    return shapley


def f_dist(dist):
    return 1/(1+dist) #We add 1 to D to avoid infinite distance

def dijkstra(start, G:nx.Graph):
    open = PriorityQueue()
    dist = {start: 0}
    increasing_order_dist = PriorityQueue()

    for v in G.nodes():
        if not v==start:
            dist[v] = np.Inf

        increasing_order_dist.add(v, dist[v])
        open.add(v, dist[v])


    while not open.is_empty():
        u = open.pop()
        for v in G.neighbors(u):
            #extract current weight between u and the current neighboor v
            w = G[u][v]["weight"]
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                increasing_order_dist.add(v, dist[v])
                #decrease priority of v
                open.add(v, alt) #If an element already exists it update the priority
    return sorted_elements(dist, increasing_order_dist)

def sorted_elements(dist, pq:PriorityQueue):
    sorted_list = []
    distances = []
    while not pq.is_empty():
        k = pq.pop()
        sorted_list.append(k)
        distances.append(dist[k])
    return distances, sorted_list

if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge('A', 'D', weight=3)
    G.add_edge('D', 'B', weight=7)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('A', 'v', weight=7)
    G.add_edge('B', 'C', weight=6)
    G.add_edge('D', 'E', weight=6)
    G.add_edge('F', 'G', weight=4)
    G.add_edge('G', 'H', weight=2)
    G.add_edge('A', 'F', weight=9)
    print(shapley_closeness(G))










