import networkx as nx
import math
import numpy as np
from scipy.sparse import linalg
import itertools as it
from joblib import Parallel, delayed
import time



def spectral_clustering(G):
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
    final_cluster={}
    final_cluster['first']=c11
    final_cluster['second']=c12
    final_cluster['third']=c21
    final_cluster['fourth']=c22
    
        
    return final_cluster


def double_chunks(data1,data2,size):
    idata1=iter(data1)
    idata2=iter(data2)
    for i in range(0, len(data1), size):
        yield [list(k) for k in it.islice(idata1, size)],[k for k in it.islice(idata2,size)]



def spectral_clustering_parallel(G,j):
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
    
    
    final_cluster={}
    final_cluster['first']=c11
    final_cluster['second']=c12
    final_cluster['third']=c21
    final_cluster['fourth']=c22
    
        
    return final_cluster


def compute_eigen(vec,nodes):
    c1=set()
    c2=set()
    for i in range(len(nodes)):
        if vec[i,0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])
    return c1,c2
