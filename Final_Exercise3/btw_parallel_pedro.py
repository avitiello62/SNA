from utils.utils import chunks
from joblib import Parallel, delayed
import math

# To automatize the decision about when an iterative algorithm of clustering like bwt_cluster must terminate,
# we can use some measure of performance of the clustering. An example of this measure is the function
# nx.algorithms.community.performance(graph, clusters). See networx documentation for more details. Given one such
# measure, one may continue iteration of the algorithm as long as the newly achieved clustering has performance that
# are not worse than the previous clustering.


def betweenness(graph, sample=None):
    """
    Function to get edge and node betweenness of a graph
    :param graph: Networkx graph
    :param sample: A subgroup of the nodes of the graph;
                  if not provided, the betweennes is calculated for all the nodes
    :return: Two dictionaries, one for edge betweenness and one for node betweenness
    """
    if sample is None:
        sample = graph.nodes()

    edge_btw = {frozenset(e): 0 for e in graph.edges()}
    node_btw = {i: 0 for i in graph.nodes()}

    for s in sample:
        # Compute the number of shortest paths from s to every other node
        tree = []  # lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in graph.nodes()}  # saves the number of shortest paths from s to i
        parents = {i: [] for i in graph.nodes()}  # saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in graph.nodes()}  # number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in graph.edges()}  # number of shortest paths starting from s that use edge e
        vflow = {i: 1 for i in graph.nodes()}  # number of shortest paths starting from s that use the vertex i.
        # It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        # BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in graph[c]:
                if distance[i] == -1:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c]+1
                if distance[i] == distance[c]+1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c = tree.pop()
            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i]/spnum[c])  # the number of shortest paths using
                # vertex c is split among the edges towards its parents proportionally
                # to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c, i})]  # each shortest path that use an edge (i,c)
                # where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c, i})]  # betweenness of an edge is the sum over
                # all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[c]  # betweenness of a vertex is the sum over all s of
                # the number of shortest paths from s to other nodes using that vertex

    return edge_btw, node_btw

def betweenness_parallel(graph, num_jobs=4):
    """
    Parallel function to get edge and node betweenness of a graph
    :param graph: Networkx graph
    :param num_jobs: Number of jobs
    :return: Two dictionaries, one for edge betweenness and one for node betweenness
    """
    edge_btw = {frozenset(e): 0 for e in graph.edges()}
    node_btw = {i: 0 for i in graph.nodes()}

    with Parallel(n_jobs=num_jobs) as parallel:
        results = parallel(delayed(betweenness)(graph, X)
                           for X in chunks(graph.nodes(), math.ceil(len(graph.nodes()) / num_jobs)))

    # Merging results
    for res in results:
        for key_edge in res[0].keys():
            edge_btw[key_edge] += res[0][key_edge]

        for key_node in res[1].keys():
            node_btw[key_node] += res[1][key_node]
    return edge_btw, node_btw