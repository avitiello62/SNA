import networkx as nx

import math

from tqdm import tqdm

import pandas as pd

from priorityq import PriorityQueue

import numpy as np
import random


def load_dataset(csv_file):

    df_edges = pd.read_csv(csv_file)

    G = nx.Graph()

    for row in tqdm(df_edges.iterrows()):

        row = row[1]

        G.add_edge(np.uint16(row["id_1"]), np.uint16(row["id_2"]) )

    print("# of self loops: ", nx.number_of_selfloops(G))

    return G


def my_hierarchical(G, num_clusters=4):
    #1. Instanziare dizionario dove la chiave è la label del nodo e il valore l'indice del cluster
    cluster_belongs = {}
    #2. Instanziare un dict dove la chiave è un insieme di nodi (frozenset e i valori sono i vicini)
    clusters = {}

    #1 Riempire clusters con tutti i nodi del grafo
    for n in G.nodes():
        clusters[frozenset([n])]=[k for k in G.neighbors(n) if not k==n] #delete self loops with the condition
        cluster_belongs[n] = frozenset([n])

    #print("Starting: ", clusters)
    #print("cluster belongs: ", cluster_belongs)
    #iteration
    condition=False
    i=0
    while len(clusters)>num_clusters:
        static_clusters = clusters.copy()
        visited = [] #collect clusters already visited in this iteration
        
        for w in static_clusters:
            if w in visited:
                continue
        

            #w = random.choice([k for k in clusters])
            #voglio unire w al cluster di uno dei suoi vicini
            #prendo un vicino random di w
            w_neighbors = clusters[w]
            w_neighbor = random.choice(w_neighbors)
            #Vedo il neighbor in che cluster si trova
            frozen_set_k_cluster = cluster_belongs[w_neighbor]
            #get neighbors of frozen_set_k
            k_neighbors = clusters[frozen_set_k_cluster]

            #delete w from clusters
            del clusters[w]
            #if frozen_set_k_cluster in clusters:
            del clusters[frozen_set_k_cluster]

            #create new frozenset with w
            new_frozen_set = frozen_set_k_cluster | w
            visited.append(frozen_set_k_cluster)
            #update with new frozenset and with neighbors
            #from w_neighbors delete elements in k neighbors
            #w_neighbors = [k for k in w_neighbors if k not in k_neighbors]
            #from k neighbors delete w
            #k_neighbors = [k for k in k_neighbors if k not in w ]
            new_neighbors = w_neighbors+k_neighbors
            new_neighbors = [k for k in set(new_neighbors) if k not in new_frozen_set]
            clusters[new_frozen_set] = new_neighbors

            #update clusters in which there is w
            for n in w: #per ogni elemento nel frozenset w
                cluster_belongs[n] = new_frozen_set

            for k in frozen_set_k_cluster:
                cluster_belongs[k] = new_frozen_set
            
            if len(clusters)==num_clusters:
                break
            #print("cluster: ", clusters)
            #print("cluster belong: ", cluster_belongs)
            #yn = input("Do you want to continue?")
            i+=1

    return clusters
def hierarchical_for_dataset(G):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in tqdm(G.nodes()):
        for v in G.nodes():
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 0)
                

    # Start with a cluster for each node
    clusters = set(frozenset([u]) for u in G.nodes())

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

        print(clusters)
        a = input("Do you want to continue? (y/n) ")
        if a == "n":
            done = True 
def two_means(G,K=4):
    n=G.number_of_nodes()
    # Choose two clusters represented by vertices that are not neighbors
    u = random.choice(list(G.nodes()))
    print(u)
    
    
    seed=[u]
    not_set=set()
    
    for i in range (K-1):
        for s in seed:
            if not_set==set():
                not_set=set(nx.non_neighbors(G, s))
            else:                
                not_set=not_set.intersection(set(nx.non_neighbors(G, s)))
        if not_set!=set():
            v=random.choice(list(not_set))
            seed.append(v)
            not_set=set()
        else:
            print("Not found seed")
            return []
    cluster0={seed[0]}
    cluster1={seed[1]}
    cluster2={seed[2]}
    cluster3={seed[3]}
    print(seed)
    added=4
    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        x = random.choice([el for el in G.nodes() if el not in cluster0|cluster1|cluster2|cluster3 and (len(
            set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0 or len(set(G.neighbors(el)).intersection(cluster2)) != 0 or len(set(G.neighbors(el)).intersection(cluster3)) != 0)])
        if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
            cluster0.add(x)
            added+=1
        elif len(set(G.neighbors(x)).intersection(cluster1)) != 0:
            cluster1.add(x)
            added+=1
        elif len(set(G.neighbors(x)).intersection(cluster2)) != 0:
            cluster2.add(x)
            added+=1
        elif len(set(G.neighbors(x)).intersection(cluster3)) != 0:
            cluster3.add(x)
            added+=1
    print(cluster0)
    print(cluster1)
    print(cluster2)
    print(cluster3)

    

    

if __name__ == '__main__':
     
    
    G = nx.Graph()
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
    G.add_edge('c', 'q')
    G.add_edge('d', 'q')
    G.add_edge('F', 'v')
    G.add_edge('F', 's')
    print("CLUSTERING")
    print("Hierarchical")
    print(two_means(G,4))