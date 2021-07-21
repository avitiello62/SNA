import networkx as nx


def get_random_graph(nodes, edges, connected=True):
    graph = nx.gnm_random_graph(nodes, edges, seed=11, directed=False)

    if connected:
        remove_small_components(graph)
    mapping = {}
    for node in graph.nodes():
        mapping[node] = str(node)
    nx.relabel_nodes(graph, mapping, False)
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
