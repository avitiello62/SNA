import networkx as nx
import math
from networkx.algorithms import cluster
from networkx.generators.directed import random_uniform_k_out_graph
from numpy.lib.type_check import real
from tqdm import tqdm
import pandas as pd
from priorityq import PriorityQueue
import numpy as np
import random
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import itertools as it
from joblib import Parallel, delayed
import time

def load_dataset(csv_file):

    df_edges = pd.read_csv(csv_file)

    G = nx.Graph()

    for row in tqdm(df_edges.iterrows()):

        row = row[1]

        G.add_edge(np.uint16(row["id_1"]), np.uint16(row["id_2"]) )

    print("# of self loops: ", nx.number_of_selfloops(G))

    return G


def load_real_cluster(csv_file):
    
    df_edges = pd.read_csv(csv_file)
    cluster={}
    cluster["tvshow"]=set()
    cluster["government"]=set()
    cluster["politician"]=set()
    cluster["company"]=set()
    for row in tqdm(df_edges.iterrows()):

        row = row[1]
        
        cluster[row["page_type"]].add(np.uint16(row["id"]))
        
    return cluster  
    
    


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
            return list(clusters)
   
def four_means(G,K=4):
    start=time.time()
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
    print("Seed Chosen")
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
        
    end=time.time()
    print("tempo esec:",end-start)
    return [cluster0 ,cluster1 ,cluster2 ,cluster3]


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
            return list(clusters)
   
def four_means_v2(G,K=4):
    start=time.time()
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
    cluster={}
    neighbors={}
    all_node=set()
    for i in range(4):
        cluster[seed[i]]=set()
        cluster[seed[i]].add(seed[i])
        neighbors[seed[i]]=set(nx.neighbors(G,seed[i]))
        all_node.add(seed[i])
    
    
    
    
    
    added=4
    while added < n:
        random_cluster=random.choice(list(cluster.keys()))
        
        if len(neighbors[random_cluster]) > 0:
            random_node=random.choice(list(neighbors[random_cluster]))
            
            cluster[random_cluster].add(random_node)
            for k in cluster.keys():
                neighbors[k].discard(random_node)
            neighbors[random_cluster]|=set(nx.neighbors(G,random_node)).difference(all_node)
            all_node.add(random_node)
            
            added+=1
    end=time.time()
    print("tempo esec:",end-start)
    
    return [cluster[key] for key in cluster]





def spectral(G):
    start=time.time()
    n=G.number_of_nodes()
    nodes=sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype() 
    w,v = linalg.eigsh(L,n-1)
    
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
    end=time.time()
    print("tempo esec:",end-start)
    return [c11, c12, c21, c22]


def double_chunks(data1,data2,size):
    idata1=iter(data1)
    idata2=iter(data2)
    for i in range(0, len(data1), size):
        yield [list(k) for k in it.islice(idata1, size)],[k for k in it.islice(idata2,size)]



def spectral_parallel(G,j):
    start=time.time()
    n=G.number_of_nodes()
    nodes=sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype() 
    w,v = linalg.eigsh(L,n-1)
    c1=set()
    c2=set() 
    
    with Parallel(n_jobs=j) as parallel:
        
        result=parallel(delayed(compute_eigen)(np.array(a),b) for a,b in double_chunks(v,nodes, math.ceil(len(G.nodes())/j)))
    #aggregrates the result
    for res in result:
        c1|=res[0]
        c2|=res[1]
    

    #starting again on c1

    n1=len(c1)
    nodes1=sorted(c1)
    L1 = nx.laplacian_matrix(G, nodes1).asfptype()
    w1,v1=linalg.eigsh(L1,n1-1)
    c11=set()
    c12=set()
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
    
        
        #print(compute_eigen(np.asarray(a),b,math.ceil(len(G.nodes())/j)))
        
        result=parallel(delayed(compute_eigen)(np.array(a),b) for a,b in double_chunks(v1,nodes1, math.ceil((n1)/j)))
        #Aggregates the results
    for res in result:
        c11|=res[0]
        c12|=res[1]

        

    #starting again on c2

    n2=len(c2)
    nodes2=sorted(c2)
    L2 = nx.laplacian_matrix(G, nodes2).asfptype()
    w2,v2=linalg.eigsh(L2,n2-1)
    c21=set()
    c22=set()
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
    
        
        #print(compute_eigen(np.asarray(a),b,math.ceil(len(G.nodes())/j)))
        
        result=parallel(delayed(compute_eigen)(np.array(a),b) for a,b in double_chunks(v2,nodes2,math.ceil((n2)/j)))
        #Aggregates the results
    for res in result:
        c21|=res[0]
        c22|=res[1]
    
    end=time.time()
    print("tempo esec:",end-start)
    return [c11,c12,c21,c22]


def compute_eigen(vec,nodes):
    c1=set()
    c2=set()
    for i in range(len(nodes)):
        if vec[i,0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])
    return c1,c2

# Computes edge and vertex betweenness of the graph in input
def betweenness(G):
    edge_btw={frozenset(e):0 for e in G.edges()}
    node_btw={i:0 for i in G.nodes()}

    for s in tqdm(G.nodes()):
        # Compute the number of shortest paths from s to every other node
        tree=[] #it lists the nodes in the order in which they are visited
        spnum={i:0 for i in G.nodes()} #it saves the number of shortest paths from s to i
        parents={i:[] for i in G.nodes()} #it saves the parents of i in each of the shortest paths from s to i
        distance={i:-1 for i in G.nodes()} 
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
    start=time.time()
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
    end=time.time()
    print("Time Exec:",end-start)
    return cc
        


def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

def betweenness_parallel(G,j=1):
    edge_btw={frozenset(e):0 for e in G.edges()}
    node_btw={i:0 for i in G.nodes()}
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        result=parallel(delayed(compute_bwt)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
    for key in edge_btw.keys():
        for res in result:
            edge_btw[key]+=res[0][key]

    for key in node_btw.keys():
        for res in result:
            node_btw[key]+=res[1][key]
    
    return edge_btw,node_btw

    
def compute_bwt(G,nodes):
    edge_btw={frozenset(e):0 for e in G.edges()}
    node_btw={i:0 for i in G.nodes()}
    for s in tqdm(nodes):
        # Compute the number of shortest paths from s to every other node
        tree=[] #it lists the nodes in the order in which they are visited
        spnum={i:0 for i in G.nodes()} #it saves the number of shortest paths from s to i
        parents={i:[] for i in G.nodes()} #it saves the parents of i in each of the shortest paths from s to i
        distance={i:-1 for i in G.nodes()} 
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

def bwt_cluster_parallel(G,j=1):
    start=time.time()
    eb,nb=betweenness_parallel(G,j)
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
    end=time.time()
    print("Time Exec:",end-start)
    return cc




if __name__ == '__main__':
    
    
    real_clusters=load_real_cluster("facebook_large/musae_facebook_target.csv")
    G=load_dataset("facebook_large/musae_facebook_edges.csv")
    four_means_clusters=four_means_v2(G)
    label=['first','second','third','fourth']
    for key in real_clusters.keys():
        cluster_len=len(real_clusters[key])
        print("Cluster {} has {} elements:".format(key,cluster_len))
        
        for i in range(4):
            intersection=len(real_clusters[key].intersection(four_means_clusters[i]))
            perc=float(intersection/len(four_means_clusters[i]))
            print("\t{} elements are in the {} cluster, the {:.2f} percentage".format(intersection,label[i],perc))
        
       
    '''
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    
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
    
    
    
    #G=load_dataset("facebook_large/musae_facebook_edges.csv")
    print("4 Means Modified")
    for i,cluster in enumerate(four_means_v2(G)):
        print("Cluster {} : {}".format(i,cluster) )
    print('Spectral Parallel')
    for i,cluster in enumerate(spectral_parallel(G,8)):  
        print("Cluster {} : {}".format(i,cluster) )
    
    print('Spectral')
    for i,cluster in enumerate(spectral(G)):  
        print("Cluster {} : {}".format(i,cluster) )
    nx.draw(G,with_labels=True)
    plt.show()
    
    
    
    
    print("Beetweenness Clustering Parallel")
    for i,cluster in enumerate(bwt_cluster_parallel(G,8)):  
        print("Cluster {} : {}".format(i,cluster) )
    print("Beetweenness Clustering")
    for i,cluster in enumerate(bwt_cluster(G)):  
        print("Cluster {} : {}".format(i,cluster) )
    
    
    
    
    
    
    
    print("CLUSTERING")
    
    print("Our Hierarchical")
    for i,cluster in enumerate(my_hierarchical(G)):
        print("Cluster {} : {}".format(i,cluster) )
    print("Hierarchical")
    for i,cluster in enumerate(hierarchical(G)):
        print("Cluster {} : {}".format(i,cluster) )
    
    G=load_dataset("facebook_large/musae_facebook_edges.csv")
    
    
    
    
    
    
    print('Spectral Parallel')
    for i,cluster in enumerate(spectral_parallel(G,8)):  
        print("Cluster {} : {}".format(i,cluster) )
    
    print("Beetweenness Clustering")
    for i,cluster in enumerate(bwt_cluster(G)):  
        print("Cluster {} : {}".format(i,cluster) )
    print('Spectral')
    for i,cluster in enumerate(spectral(G)):  
        print("Cluster {} : {}".format(i,cluster) )
    
    '''
    