from src.betweenness import *
from src.four_means import *
from src.hierarchical import *
from src.spectral import *
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

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
    
    G=load_dataset("../facebook_large/musae_facebook_edges.csv")
    
    
    
    
    
    
    
    print('BTW Parallel ')#40min    5/10 volte     alberto
    clusters=btw_clustering_parallel(G,8)
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'../btw_parallel2.pkl')


    
    