import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import networkx as nx
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


