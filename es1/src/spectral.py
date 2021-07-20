import networkx as nx
import math
import numpy as np
from scipy.sparse import linalg
import itertools as it
from joblib import Parallel, delayed
import time


def compute_eigen(vec,nodes):
    c1=set()
    c2=set()
    for i in range(len(nodes)):
        if vec[i,0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])
    return c1,c2

def double_chunks(data1,data2,size):
    idata1=iter(data1)
    idata2=iter(data2)
    for i in range(0, len(data1), size):
        yield [list(k) for k in it.islice(idata1, size)],[k for k in it.islice(idata2,size)]

def spectral_two_clustering(G,nodes,len):
    #compute the laplacian matrix
    L = nx.laplacian_matrix(G, nodes).asfptype()
    #compute the eigen values and eigen vectors from the matrix 
    w,v = linalg.eigsh(L,len-1)
    #we start clusterizing the nodes in two clusters
    cluster1,cluster2=compute_eigen(v,nodes)
    return cluster1, cluster2

def parallel_spectral_two_clustering(G,nodes,n,j):
    #compute the laplacian matrix
    L = nx.laplacian_matrix(G, nodes).asfptype()
    #compute the eigen values and eigen vectors from the matrix 
    w,v = linalg.eigsh(L,n-1)
    #we start clusterizing the nodes in two clusters
    #we start clusterizing the nodes in two clusters
    cluster1=set()
    cluster2=set() 
    #we split the eigen vectors check among the different jobs
    with Parallel(n_jobs=j) as parallel:    
        result=parallel(delayed(compute_eigen)(np.array(a),b) for a,b in double_chunks(v,nodes, math.ceil(n/j)))
    #now it is necessary to aggregate  the results
    for res in result:
        cluster1|=res[0]
        cluster2|=res[1]
    return cluster1, cluster2

def spectral_clustering(G):
    # saving starting time of the algorithm
    start = time.time()
    n=G.number_of_nodes()
    nodes=sorted(G.nodes())
    
    #we start clusterizing the nodes in two clusters
    clusters=[]
    result=spectral_two_clustering(G,nodes,n)
    #we need to resplit each cluster into two clusters, in order to make  4 clusters
    
    #first split
    n1=len(result[0])
    nodes1=sorted(result[0])
    result2=spectral_two_clustering(G,nodes1,n1)
    clusters.append(result2[0])
    clusters.append(result2[1])
    #second split
    n2=len(result[1])
    nodes2=sorted(result[1])
    result2=spectral_two_clustering(G,nodes2,n2)
    clusters.append(result2[0])
    clusters.append(result2[1])
    
    end=time.time()
    # algorithm execution time
    print("Execution time:", end-start)
    # we format the output into a dict
    final_cluster={}
    final_cluster['first']=clusters[0]
    final_cluster['second']=clusters[1]
    final_cluster['third']=clusters[2]
    final_cluster['fourth']=clusters[3]
        
    return final_cluster

def spectral_clustering_parallel(G,j):
    #saving starting time of the algorithm
    start=time.time()
    n=G.number_of_nodes()
    nodes=sorted(G.nodes())
    
    #we start clusterizing the nodes in two clusters
    clusters=[]
    result=parallel_spectral_two_clustering(G,nodes,n,j)
    

    #first split
    n1=len(result[0])
    nodes1=sorted(result[0])
    result2=parallel_spectral_two_clustering(G,nodes1,n1,j)
    clusters.append(result2[0])
    clusters.append(result2[1])
        

    #second split
    n2=len(result[1])
    nodes2=sorted(result[1])
    result2=parallel_spectral_two_clustering(G,nodes2,n2,j)
    clusters.append(result2[0])
    clusters.append(result2[1])
    
    end=time.time()
    # algorithm execution time
    print("Execution time:", end-start)
    # we format the output into a dict
    final_cluster={}
    final_cluster['first']=clusters[0]
    final_cluster['second']=clusters[1]
    final_cluster['third']=clusters[2]
    final_cluster['fourth']=clusters[3]
        
    return final_cluster


