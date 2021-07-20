import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import sys
#local import (running the python command in the src folder)
from betweenness import *
from four_means import *
from hierarchical import *
from spectral import *
sys.path.append('../')
from utils.priorityq import PriorityQueue
from utils.es1_utils import *


if __name__ == '__main__':    
    
    #loading the real clusters from file
    real_clusters=load_real_cluster("../../facebook_large/musae_facebook_target.csv")
    #setting for output on file
    label=['first','second','third','fourth']
    f = open("../results/output.txt", "w")
    #setting to load an algorithm's ouput
    name="../results/spectral_sampled08"+".pkl"
    output_clusters=load_dict_from_file(name)
    #starting a pq in order to sort the clusters common element percentage
    pq=PriorityQueue()
    #just for file ouput
    f.write("\n"+name+"\n")
    f.write("----------------------------------------------------------\n\n")
    #for each output cluster
    for k in label:
        cluster_len=len(output_clusters[k])
        #some statics
        print("Cluster {} has {} elements:".format(k,cluster_len))
        #for each real cluster
        for key in sorted(real_clusters.keys()):
            #count the common element between the real cluster[key] and our_cluster[k]
            intersection=len(real_clusters[key].intersection(output_clusters[k]))
            #compute the percentage over the output cluster number of elements
            perc=float(intersection/cluster_len)
            print("\t{} elements are in the {} cluster, the {:.2f} percentage".format(intersection,key,perc))
            #we order the elements for this percentage
            value=tuple([key,k])
            pq.add(value,-perc)
    
    used_clusters=[]
    used_real_clusters=[]
    ass={} 
    ass_prob={}
    try:
        while(True):
            el,priority=pq.pop_adv()
            #if the current real cluster or the output cluster is already assigned
            if el[0] in used_real_clusters or el[1] in used_clusters:
                continue
            #we assign the two cluster each other
            used_clusters.append(el[1])
            used_real_clusters.append(el[0])
            ass[el[0]]=el[1]
            ass_prob[el[0]]=-priority
    except:
        pass
    #for each tuple (real cluster, our cluster) we compute the percentage of elements rightly classified
    for key in sorted(ass.keys()):
        string=key+": "+ass[key]+"\n"
        f.write(string)
        string2=key+": "+str(ass_prob[key])+"\n"
        f.write(string2)
        intersection=len(real_clusters[key].intersection(output_clusters[ass[key]]))
        perc=float(intersection/len(real_clusters[key]))
        print("\t{} elements of {} , the {:.2f} percentage".format(intersection,len(real_clusters[key]),perc))
        f.write("\t{} elements of {} , the {:.2f} percentage".format(intersection,len(real_clusters[key]),perc))
        f.write("\n")
    f.close()
    