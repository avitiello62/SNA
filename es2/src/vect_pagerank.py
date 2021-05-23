import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from networkx.algorithms.link_analysis.pagerank_alg import pagerank, pagerank_numpy

def load_dataset(csv_file):
    df_edges = pd.read_csv(csv_file)
    G = nx.Graph()
    for row in tqdm(df_edges.iterrows()):
        row = row[1]
        G.add_edge(np.uint16(row["id_1"]), np.uint16(row["id_2"]) )
    print("# of self loops: ", nx.number_of_selfloops(G))
    print("# nodes: ", len(G))
    return G


def vectorized_pagerank(G, d=0.85, tol=1.0e-11, max_iter=100, dense=False):
    #Transition matrix M
    #rank vector (initially 1/n)
    #update: v' = Mv
    

    G.remove_edges_from(nx.selfloop_edges(G))
    M = nx.adjacency_matrix(G)
    M=M.astype(np.float64)    
    N = len(G)
    v = np.ones(len(G))/N
    error_matrix = tol * np.ones(N)
    if dense:
        M=M.todense()
        v = np.ones((N, 1) )/N
    
    for n in G.nodes:
        divisor = len([k for k in G.neighbors(n)])

        if divisor>0:
            M[n,:] = M[n,:]/divisor
    
    #print((M>1).sum()>0)
    L=M.T

    i=0

    while i<max_iter:
        v_old = v
        v = L*v_old
        if dense:
            v = np.dot(M,v_old)

        i+=1
        v = v/max(v)
        err = np.absolute(v - v_old).sum()
        if err<N*tol:
            return v

        print("{}th it. ".format(i))
        #print("Max 1?: ", (v>1).sum())
        #print(v)        
    
    return v


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(3, 4)
    G.add_edge(3, 5)
    G.add_edge(3, 6)
    G.add_edge(4, 5)
    G.add_edge(5, 6)
    G=load_dataset("facebook_large\\musae_facebook_edges.csv")
    import time
    print("vectorized pagerank")
    start = time.time()
    v = vectorized_pagerank(G, dense=False)
    end = time.time()
    print("Time: ", end-start)
    print(v)
    #v = eig_pagerank(G)

    #v = pagerank_numpy(G)
    #print(v)
    N = len(G)
    print("networkx pagerank")
    start = time.time()
    v = pagerank(G)
    end = time.time()
    print("Time: ", end-start)
    print(v[0], v[1], v[2],"...",v[N-3], v[N-2], v[N-1])



    

    
