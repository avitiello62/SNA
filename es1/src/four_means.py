
import networkx as nx
import random
import time


def four_means_clustering(G,K=4):
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
    final_cluster={}
    final_cluster['first']=cluster0
    final_cluster['second']=cluster1
    final_cluster['third']=cluster2
    final_cluster['fourth']=cluster3
    return final_cluster
   
def four_means_clustering_opt(G,K=4):
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
    
    

    label=['first','second','third','fourth']
    final_cluster={}
    i=0
    for k in cluster:
        final_cluster[label[i]]=cluster[k]
        i+=1
    return final_cluster


