import numpy as np
import time
import networkx as nx
from tqdm import tqdm


# Naive HITS


def hits(G: nx.Graph, epsilon=1e-8, normalized=True, max_it=500):
    nodes = G.nodes
    auth = {}
    hub = {}

    # Initialization phase
    for node in nodes:
        auth[node] = 1.0
        hub[node] = 1.0

    for i in tqdm(range(max_it)):
        hits_one_iter(G, auth, hub)

    return hub


def hits_one_iter(G, auth, hub):
    node_list = G.nodes

    for node in node_list:
        update_auth(G, node, auth, hub)

    for node in node_list:
        update_hub(G, node, auth, hub)

    normalize_auth_hub(G, auth, hub)


def get_auth_hub_list(G):
    auth_list = np.asarray([node.auth for node in G.nodes], dtype='float32')
    hub_list = np.asarray([node.hub for node in G.nodes], dtype='float32')

    return np.round(auth_list, 3), np.round(hub_list, 3)


def update_auth(G: nx.Graph, node, auth, hub):
    auth[node] = sum(hub[node] for node in G[node])


def update_hub(G: nx.Graph, node, auth, hub):
    hub[node] = sum(auth[node] for node in G[node])


def normalize_auth_hub(G: nx.Graph, auth, hub):
    nodes = G.nodes
    auth_sum = sum(auth[node] for node in nodes)
    hub_sum = sum(hub[node] for node in nodes)

    for node in nodes:
        auth[node] /= auth_sum
        hub[node] /= hub_sum


def my_hits(G, epsilon=1e-8, normalized=True, max_it=50):
    L = nx.adjacency_matrix(G)
    L_T = L.transpose()
    n = L.shape[0]
    epsilon_matrix = epsilon * np.ones(n)
    h = np.ones(n, dtype=np.float64)
    a = np.ones(n, dtype=np.float64)
    it = 0
    while it < max_it:
        a_old = a
        h_old = h

        a = L_T * h_old
        a_max = a.max(axis=0)
        a = a / a_max if a_max > 0 else a
        h = L * a
        h_max = h.max(axis=0)
        h = h / h_max if h_max > 0 else h

        # print("h: ", h)
        # print("a: ", a)
        # input("Continue?")

        if (((abs(h - h_old)) < epsilon_matrix).all()) and (((abs(a - a_old)) < epsilon_matrix).all()):
            break
        it += 1

    return h


if __name__ == "__main__":
    from es1.main import load_dataset

    G = load_dataset("../facebook_large/musae_facebook_edges.csv")
    h, a = hits(G)
