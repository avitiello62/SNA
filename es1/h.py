import networkx as nx
import math
from tqdm import tqdm
import pandas as pd
from priorityq import PriorityQueue
import numpy as np
import random
from scipy.sparse import linalg
import matplotlib.pyplot as plt

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

    return [ set(k) for k in clusters]
           
def four_means(G,K=4):
    n=G.number_of_nodes()
    u = random.choice(list(G.nodes()))
    seed=[u]
    not_neighbors_set=set()
    
    for i in range (K-1):
        for s in seed:
            if not_neighbors_set==set():
                not_neighbors_set=set(nx.non_neighbors(G, s))
            else:                
                not_neighbors_set=not_neighbors_set.intersection(set(nx.non_neighbors(G, s)))
        if not_neighbors_set!=set():
            v=random.choice(list(not_neighbors_set))
            seed.append(v)
            not_neighbors_set=set()
        else:
            print("Not found seed")
            return []
    cluster0={seed[0]}
    cluster1={seed[1]}
    cluster2={seed[2]}
    cluster3={seed[3]}
    
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
    return [cluster0 ,cluster1 ,cluster2 ,cluster3]

def spectral(G):
    n=G.number_of_nodes()
    nodes=sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype() #Laplacian of a graph is a matrix, with diagonal entries being the degree of the corresponding node and off-diagonal entries being -1 if an edge between the corresponding nodes exists and 0 otherwise
    #print(L) #To see the laplacian of G uncomment this line
    # The following command computes eigenvalues and eigenvectors of the Laplacian matrix.
    # Recall that these are scalar numbers w_1, ..., w_k and vectors v_1, ..., v_k such that Lv_i=w_iv_i.
    # The first output is the array of eigenvalues in increasing order. The second output contains the matrix of eigenvectors: specifically, the eigenvector of the k-th eigenvalue is given by the k-th column of v
    w,v = linalg.eigsh(L,n-1)
    #print(w) #Print the list of eigenvalues
    #print(v) #Print the matrix of eigenvectors
    #print(v[:,0]) #Print the eigenvector corresponding to the first returned eigenvalue

    # Partition in clusters based on the corresponding eigenvector value being positive or negative
    # This is known to return (an approximation of) the sparset cut of the graph
    # That is, the cut with each of the clusters having many edges, and with few edge among clusters
    # Note that this is not the minimum cut (that only requires few edge among clusters, but it does not require many edge within clusters)
    c1=set()
    c2=set()
    for i in range(n):
        if v[i,0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])
    #we need to resplit the two cluster in 2 part

    n1=len(c1)
    nodes1=sorted(c1)
    L1 = nx.laplacian_matrix(G, nodes1).asfptype()
    w1,v1=linalg.eigsh(L1,n1-1)
    c11=set()
    c12=set()
    for i in range(n1):
        if v1[i,0] < 0:
            c11.add(nodes1[i])
        else:
            c12.add(nodes1[i])
    #print(c11,c12)
    
    n2=len(c2)
    nodes2=sorted(c2)
    L2 = nx.laplacian_matrix(G, nodes2).asfptype()
    w2,v2=linalg.eigsh(L2,n2-1)
    c21=set()
    c22=set()
    for i in range(n2):
        if v2[i,0] < 0:
            c21.add(nodes2[i])
        else:
            c22.add(nodes2[i])
    #print(c21,c22)
    return [c11, c12, c21, c22]

# Computes edge and vertex betweenness of the graph in input
def betweenness(G):
    edge_btw={frozenset(e):0 for e in G.edges()}
    node_btw={i:0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree=[] #it lists the nodes in the order in which they are visited
        spnum={i:0 for i in G.nodes()} #it saves the number of shortest paths from s to i
        parents={i:[] for i in G.nodes()} #it saves the parents of i in each of the shortest paths from s to i
        distance={i:-1 for i in G.nodes()} #the number of shortest paths starting from s that use the edge e
        eflow={frozenset(e):0 for e in G.edges()} #the number of shortest paths starting from s that use the edge e
        vflow={i:1 for i in G.nodes()} #the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue=[s]
        spnum[s]=1
        distance[s]=0
        while queue != []:
            c=queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1: #if vertex i has not been visited
                    queue.append(i)
                    distance[i]=distance[c]+1
                if distance[i] == distance[c]+1: #if we have just found another shortest path from s to i
                    spnum[i]+=spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c=tree.pop()
            for i in parents[c]:
                eflow[frozenset({c,i})]+=vflow[c] * (spnum[i]/spnum[c]) #the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i]+=eflow[frozenset({c,i})] #each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c,i})]+=eflow[frozenset({c,i})] #betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c]+=vflow[c] #betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw,node_btw

#The algorithm is quite time-consuming. Indeed, its computational complexity is O(nm).
#Possible optimizations: parallelization, considering only a sample of starting nodes

#Clusters are computed by iteratively removing edges of largest betweenness
def bwt_cluster(G):
    eb,nb=betweenness(G)
    pq=PriorityQueue()
    for i in eb.keys():
        pq.add(i,-eb[i])
    graph=G.copy()
    #we can stop the algorithm when there are only 4 cluster (connected component in the graph)
    cc=[]
    while len(cc)!=4:
        edge=tuple(sorted(pq.pop()))
        graph.remove_edges_from([edge])
        cc=list(nx.connected_components(graph))
    return cc
        

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
    G.add_edge('P', 'q')
    G.add_edge('d', 'q')
    G.add_edge('F', 'V')
    G.add_edge('F', 's')
    G.add_edge('V', '1')
    G.add_edge('1', '2')
    G.add_edge('2', '3')
    G.add_edge('3', '1')



    print("CLUSTERING")
    
    print("Our Hierarchical")
    for i,cluster in enumerate(my_hierarchical(G)):
        print("Cluster {} : {}".format(i,cluster) )
    
    print("4 Means")
    for i,cluster in enumerate(four_means(G)):
        print("Cluster {} : {}".format(i,cluster) )
    
    print("Spectral")
    for i,cluster in enumerate(spectral(G)): 
        print("Cluster {} : {}".format(i,cluster) )
    print("Beetweenness Clustering")
    for i,cluster in enumerate(bwt_cluster(G)):  
        print("Cluster {} : {}".format(i,cluster) )
    
    nx.draw(G,with_labels=True)
    plt.show()
    
    