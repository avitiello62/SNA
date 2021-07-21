import networkx as nx


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
