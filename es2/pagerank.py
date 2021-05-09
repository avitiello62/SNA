import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import copy
def pagerank(G: nx.DiGraph):
    print("gg")


def directed_graph_generation(n:int, p:float):
    return nx.generators.random_graphs.fast_gnp_random_graph(n, p, directed=True)

def plot_function(G):
    pos = nx.layout.spring_layout(G)

    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="blue")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    n = 1000
    p = 0.1
    tollerance = 1e-6
    G = directed_graph_generation(n, p)
    print(len(G.edges), len(G.nodes))

    edge_rank = {}
    node_rank = {}
    for node in G.nodes:
        node_rank[node] = 1/n
        #G.nodes[x]["weight"] = 1 / n
        #print(G.nodes[x]["weight"])



    h = 0
    b = {}
    rank = 0

    #nx.draw(G)
    #plt.show()
    while True:
        node_rank_old = copy.deepcopy(node_rank)
        edge_rank_old = copy.deepcopy(edge_rank)
        for node in G.nodes:
            neighbors = [k for k in G.neighbors(node)]
            divisor = len(list(neighbors))
            for neigh in neighbors:
                edge_rank[frozenset([node, neigh])] = node_rank[node]/divisor

        for node in G.nodes:
            node_rank[node] = 0
            neighbors = [k for k in G.neighbors(node)]
            for neigh in neighbors:
                node_rank[node] += edge_rank[frozenset([node, neigh])]

        h += 1

        err_node_rank = sum([abs(node_rank[n] - node_rank_old[n]) for n in node_rank])
        if err_node_rank < n*tollerance:
            break

    print(len(node_rank), node_rank)
    print(len(edge_rank), edge_rank)
    print("CONVERGENZA = ", h)



    '''pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    plt.show()'''

    ''' for edge in G.edges:
        print(edge)
        if edge[0] in b.keys():
            b[edge[0]] += 1
            rank += 1
        else:
            b[edge[0]] = 1
            rank += 1

        if edge[0] > h:
            h += 1'''


