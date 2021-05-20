import random


def hierarchical_clustering_opt(G, num_clusters=4):
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
            for n in w: #per ogni elemento nel frozenset w
                cluster_belongs[n] = new_frozen_set

            for k in neighbor_cluster:
                cluster_belongs[k] = new_frozen_set
            
            if len(clusters)==num_clusters:
                break
            #print("cluster: ", clusters)
            #print("cluster belong: ", cluster_belongs)
            #yn = input("Do you want to continue?")
            i+=1
    label=['first','second','third','fourth']
    final_cluster={}
    i=0
    for k in clusters:
        final_cluster[label[i]]=set(k)
        i+=1
    return final_cluster

  