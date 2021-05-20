from src.betweenness import *
from src.four_means import *
from src.hierarchical import *
from src.spectral import *
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    
    
def save_dict_on_file(dict_data,file_name):
    a_file = open(file_name, "wb")
    pickle.dump(dict_data, a_file)
    a_file.close()
    return

def load_dict_from_file(file_name):
    a_file = open(file_name, "rb")
    output = pickle.load(a_file)
    return output



if __name__ == '__main__':
    
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
    
    
    print("4 Means OPT")
    clusters=four_means_clustering_opt(G)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'four_means_opt.pkl')
    
    print("4 Means ")
    clusters=four_means_clustering(G)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'four_means.pkl')
    
    
    print('Spectral Parallel')
    clusters=spectral_clustering_parallel(G,4)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'spectral_parallel.pkl')
    
    
    print('Spectral ')
    clusters=spectral_clustering(G)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'spectral.pkl')
    
    
    print('Hierarchical OPT ')
    clusters=hierarchical_clustering_opt(G,4)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'hierarchical_opt.pkl')
    
    
    print('BTW Parallel ')
    clusters=btw_clustering_parallel(G,4)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'btw_parallel.pkl')


    print('BTW  ')
    clusters=btw_clustering(G)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'btw.pkl')
    