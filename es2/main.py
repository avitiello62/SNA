from es1.main import *
from utils.lesson2 import *
from utils.priorityq import PriorityQueue


def top_new(G, measure, k):
    pq = PriorityQueue()
    cen = measure(G)
    for u in G.nodes():
        pq.add(u, -cen[
            u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    out = []
    for i in range(k):
        out.append(pq.pop())
    return out


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')

    G.add_edge('A', 'B')
    G.add_edge('A', 'v')
    G.add_edge('B', 'C')

    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'z')

    G.add_edge('E', 'F')
    G.add_edge('E', 's')
    G.add_edge('F', 'G')
    G.add_edge('F', 'q')
    G.add_edge('P', 'q')
    G.add_edge('d', 'q')
    G.add_edge('F', 'V')
    G.add_edge('F', 's')
    G.add_edge('V', '1')
    G.add_edge('1', '2')
    G.add_edge('2', '3')
    G.add_edge('3', '1')

    G = load_dataset("facebook_large/musae_facebook_edges.csv")

    top_number = 500
    print("Degree")
    rank = top(G, degree, top_number)
    for k in clusters:
        print("Cluster {} : {}".format(k, clusters[k]))
    save_dict_on_file(clusters, 'four_means_opt.pkl')
