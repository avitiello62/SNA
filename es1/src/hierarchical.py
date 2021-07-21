import random
from utils.priorityq import PriorityQueue

def hierarchical(G):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 0)
                else:
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 1)

    # Start with a cluster for each node
    clusters = set(frozenset(u) for u in G.nodes())

    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)

        clusters.add(s[0] | s[1])

        if len(clusters)==4:
            done = True


def hierarchical_clustering_opt(G, num_clusters=4):
    #1. instantiate a dictionary where the key is the node’s label and the value the index of cluster the node belongs to
    cluster_belongs = {}
    #2. Instantiate a dict where the key is a set of nodes(frozenset) and the value will be the neighbours
    clusters = {}

    #1 Fill cluster with all nodes of network
    for n in G.nodes():
        clusters[frozenset([n])]=[k for k in G.neighbors(n) if not k==n] #delete self loops with the condition
        cluster_belongs[n] = frozenset([n])

    i=0
    while len(clusters)>num_clusters:
        static_clusters = clusters.copy()
        visited = [] #collect clusters already visited in this iteration
        
        for w in static_clusters:
            if w in visited:
                continue

            #I want to merge w to one of the neighbours' cluster
            #take a random neighbour of w
            w_neighbors = clusters[w]
            w_neighbor = random.choice(w_neighbors)
            #Pick up neighbour's cluster
            neighbor_cluster = cluster_belongs[w_neighbor]
            #get neighbors of neighbor
            k_neighbors = clusters[neighbor_cluster]

            #delete w from clusters
            del clusters[w]
            #if neighbor_cluster in clusters:
            del clusters[neighbor_cluster]

            #create new frozenset with w
            new_frozen_set = neighbor_cluster | w
            visited.append(neighbor_cluster)
            #update with new frozenset and with neighbors
            #from w_neighbors delete elements in k neighbors
            #w_neighbors = [k for k in w_neighbors if k not in k_neighbors]
            #from k neighbors delete w
            #k_neighbors = [k for k in k_neighbors if k not in w ]
            new_neighbors = w_neighbors+k_neighbors
            new_neighbors = [k for k in set(new_neighbors) if k not in new_frozen_set]
            clusters[new_frozen_set] = new_neighbors

            #update clusters in which there is w
            for n in w: #for each element in the frozenset w
                cluster_belongs[n] = new_frozen_set

            for k in neighbor_cluster:
                cluster_belongs[k] = new_frozen_set
            
            if len(clusters)==num_clusters:
                break
            i+=1
    label=['first','second','third','fourth']
    final_cluster={}
    i=0
    for k in clusters:
        final_cluster[label[i]]=set(k)
        i+=1
    return final_cluster

  