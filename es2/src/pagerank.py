from pathlib import Path
import networkx as nx
import copy
import time


def pagerank_naive(G):
    tollerance = 1e-6
    n = len(G.nodes())
    h = 0
    edge_rank = {}
    node_rank = {}

    start_time = time.time()

    for node in G.nodes:
        node_rank[node] = 1 / n

    while True:
        node_rank_old = copy.deepcopy(node_rank)
        edge_rank_old = copy.deepcopy(edge_rank)
        for node in G.nodes:
            neighbors = [k for k in G.neighbors(node)]
            divisor = len(list(neighbors))
            for neigh in neighbors:
                edge_rank[frozenset([node, neigh])] = node_rank[node] / divisor

        for node in G.nodes:
            node_rank[node] = 0
            neighbors = [k for k in G.neighbors(node)]
            for neigh in neighbors:
                node_rank[node] += edge_rank[frozenset([node, neigh])]

        h += 1

        err_node_rank = sum([abs(node_rank[n] - node_rank_old[n]) for n in node_rank])

        if err_node_rank < n * tollerance:
            stop_time = time.time()
            print("ERRORE: ", err_node_rank)
            print("TOLLERANZA DEL GRAFO: ", tollerance)
            print("ITERAZIONI PER LA CONVERGENZA = ", h)
            print("TEMPO = ", stop_time - start_time)
            break

    return node_rank


def load_graph(csv_file):
    base_path = Path(__file__).parent
    file_path = (base_path / csv_file).resolve()
    data = open(file_path, "r")
    next(data, None)  # skip the first line in the input file
    Graphtype = nx.Graph()

    G = nx.parse_edgelist(data, delimiter=',', create_using=Graphtype,
                          nodetype=int, data=(('weight', float),))

    return G


if __name__ == '__main__':
    n = 22470
    p = 0.00068

    G = load_graph(csv_file="../../musae_facebook_edges.csv")

    node_rank = pagerank_naive(G)

    # print(node_rank) #chiave = nodo valore = rank
