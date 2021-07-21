import networkx as nx
import random
import time


def four_means_clustering(G):
    K = 4
    # saving starting time of the algorithm
    start = time.time()
    n = G.number_of_nodes()
    # random choice the first seed (it will be the first cluster)
    u = random.choice(list(G.nodes()))
    seed = [u]
    not_neighbors_set = set()
    # repeat the choice k-1 times
    for i in range(K-1):
        # for each already choosen seed, we check all the not neighbors in order to make the next choice
        for s in seed:
            if not_neighbors_set == set():
                not_neighbors_set = set(nx.non_neighbors(G, s))
            else:
                # if we have more seeds, we need to intersect the notneighbors
                not_neighbors_set = not_neighbors_set.intersection(
                    set(nx.non_neighbors(G, s)))
        # if we got st least one element, we can choice the next seed
        if not_neighbors_set != set():
            v = random.choice(list(not_neighbors_set))
            seed.append(v)
            not_neighbors_set = set()
        else:
            # is not possible to run the algorithm
            print("Not found seed")
            return []
    cluster0 = {seed[0]}
    cluster1 = {seed[1]}
    cluster2 = {seed[2]}
    cluster3 = {seed[3]}
    added = 4
    while added < n:
        # Choose a node that is not yet in a cluster but with an edge in common with a node already clusterized and add it in this cluster
        x = random.choice([el for el in G.nodes() if el not in cluster0 | cluster1 | cluster2 | cluster3 and (len(
            set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0 or len(set(G.neighbors(el)).intersection(cluster2)) != 0 or len(set(G.neighbors(el)).intersection(cluster3)) != 0)])
        if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
            cluster0.add(x)
            added += 1
        elif len(set(G.neighbors(x)).intersection(cluster1)) != 0:
            cluster1.add(x)
            added += 1
        elif len(set(G.neighbors(x)).intersection(cluster2)) != 0:
            cluster2.add(x)
            added += 1
        elif len(set(G.neighbors(x)).intersection(cluster3)) != 0:
            cluster3.add(x)
            added += 1
    end = time.time()
    # algorithm execution time
    print("Execution time:", end-start)
    # we format the output into a dict
    final_cluster = {}
    final_cluster['first'] = cluster0
    final_cluster['second'] = cluster1
    final_cluster['third'] = cluster2
    final_cluster['fourth'] = cluster3
    return final_cluster


def four_means_clustering_opt(G):
    K = 4
    # saving starting time of the algorithm
    start = time.time()
    n = G.number_of_nodes()
    # random choice the first seed (it will be the first cluster)
    u = random.choice(list(G.nodes()))
    seed = [u]
    not_neighbors_set = set()
    # repeat the choice k-1 times
    for i in range(K-1):
        # for each already choosen seed, we check all the not neighbors in order to make the next choice
        for s in seed:
            if not_neighbors_set == set():
                not_neighbors_set = set(nx.non_neighbors(G, s))
            else:
                # if we have more seeds, we need to intersect the notneighbors
                not_neighbors_set = not_neighbors_set.intersection(
                    set(nx.non_neighbors(G, s)))
        # if we got st least one element, we can choice the next seed
        if not_neighbors_set != set():
            v = random.choice(list(not_neighbors_set))
            seed.append(v)
            not_neighbors_set = set()
        else:
            # is not possible to run the algorithm
            print("Not found seed")
            return []

    clusters = {}
    neighbors = {}
    all_node = set()
    for i in range(4):
        # clusters key is the first node in the cluster
        clusters[seed[i]] = set()
        clusters[seed[i]].add(seed[i])
        # neighbors contains the neighbors of all the node in a cluster, the key is also here the first node
        neighbors[seed[i]] = set(nx.neighbors(G, seed[i]))
        # tracking all the node already clusterized
        all_node.add(seed[i])

    added = K
    while added < n:
        # randomly choosing a cluster key
        random_cluster = random.choice(list(clusters.keys()))
        # if there is at least a neighbor of a node in that cluster
        if len(neighbors[random_cluster]) > 0:
            # we choose a random neighbor of the cluster
            random_node = random.choice(list(neighbors[random_cluster]))
            # we add it to the cluster
            clusters[random_cluster].add(random_node)
            # we discard it from all the cluster's neighbor (it will be not more considered for the random pick, is already clusterized)
            for k in clusters.keys():
                neighbors[k].discard(random_node)
            # we update the cluster neighbors (not including the already picked node)
            neighbors[random_cluster] |= set(
                nx.neighbors(G, random_node)).difference(all_node)
            # track also this node
            all_node.add(random_node)
            added += 1

    end = time.time()
    # algorithm execution time
    print("Execution time:", end-start)

    # we format the output into a dict
    label = ['first', 'second', 'third', 'fourth']
    final_cluster = {}
    i = 0
    for k in clusters:
        final_cluster[label[i]] = clusters[k]
        i += 1
    return final_cluster
