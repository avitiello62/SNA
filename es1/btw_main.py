from networkx.algorithms import cluster
from src.betweenness import *
from src.four_means import *
from src.hierarchical import *
from src.spectral import *
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

sys.path.append('../utils')
from priorityq import PriorityQueue

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
    
    #G = nx.Graph()
    
    
    #G=nx.gnp_random_graph(20,0.50,seed=11)
    
    
    

    
    
    
    


   
    #a valle del clustering, per ogni cluster contare la frequenza delle varie classi vere e assegnare la percentuale piu alta ad ognuno
    
    real_clusters=load_real_cluster("../facebook_large/musae_facebook_target.csv")
    #G=load_dataset("../facebook_large/musae_facebook_edges.csv")
    
    label=['first','second','third','fourth']
    f = open("results/demofile10.txt", "w")
    
    for i in range(1):
        pq=PriorityQueue()
        name="spectral_sampled08"+".pkl"
        four_means_clusters=load_dict_from_file(name)
        f.write("\n"+name+"\n")
        f.write("----------------------------------------------------------\n\n")
        for k in label:
            cluster_len=len(four_means_clusters[k])
            print("Cluster {} has {} elements:".format(k,cluster_len))
            
            for key in sorted(real_clusters.keys()):
                intersection=len(real_clusters[key].intersection(four_means_clusters[k]))
                perc=float(intersection/cluster_len)
                print("\t{} elements are in the {} cluster, the {:.2f} percentage".format(intersection,key,perc))
                value=tuple([key,k])
                pq.add(value,-perc)
        used_clusters=[]
        used_real_clusters=[]
        ass={} 
        ass_prob={}
        try:
            while(True):
                el,priority=pq.pop_adv()
                if el[0] in used_real_clusters or el[1] in used_clusters:
                    
                    continue
                used_clusters.append(el[1])
                used_real_clusters.append(el[0])
                ass[el[0]]=el[1]
                ass_prob[el[0]]=-priority
        except:
            pass
        
        for key in sorted(ass.keys()):
            string=key+": "+ass[key]+"\n"
            f.write(string)
            string2=key+": "+str(ass_prob[key])+"\n"
            f.write(string2)
            intersection=len(real_clusters[key].intersection(four_means_clusters[ass[key]]))
            perc=float(intersection/len(real_clusters[key]))
            print("\t{} elements of {} , the {:.2f} percentage".format(intersection,len(real_clusters[key]),perc))
            f.write("\t{} elements of {} , the {:.2f} percentage".format(intersection,len(real_clusters[key]),perc))
            f.write("\n")
    f.close()
    '''
    name="spectral_sampled_08"+".pkl"
    four_means_clusters=load_dict_from_file(name)
    real_fucking_cluster={}
    label=['first','second','third','fourth']
    for i in range(4):
        real_fucking_cluster[label[i]]=four_means_clusters[i]
    save_dict_on_file(real_fucking_cluster,'spectral_sampled08.pkl')
    print(real_fucking_cluster)
    '''