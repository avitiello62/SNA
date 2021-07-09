import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed
from tqdm import tqdm


# The measure associated to each node is the sum of the (shortest) distances of this node from each remaining node

def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


def closeness(G, sample=None):
    cen = dict()
    if sample is None:
        sample = G.nodes()

    for i, u in enumerate(sample):
        if (i + 1) == 2000:
            print(i)
        visited = set()
        visited.add(u)
        queue = [u]
        dist = dict()
        dist[u] = 0

        while len(queue) > 0:
            v = queue.pop(0)
            for w in G[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    dist[w] = dist[v] + 1

        cen[u] = sum(dist.values())

    return cen


def parallel_closeness(G, j=8):
    cen = {}
    with Parallel(n_jobs=j) as parallel:
        # Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        result = parallel(delayed(closeness)(G, X) for X in chunks(G.nodes(), math.ceil(len(G.nodes()) / j)))
        for res in result:
            cen.update(res)
    return cen
