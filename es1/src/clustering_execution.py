import sys

#local import (running the python command in the src folder)
from betweenness import *
from four_means import *
from hierarchical import *
from spectral import *
sys.path.append('../')
from utils.es1_utils import *

if __name__ == '__main__':
    #execution example
    G=load_dataset("../../facebook_large/musae_facebook_edges.csv")
    clusters=betwenness_clustering(G)   
    for k in clusters:
        print("Cluster {} : {}".format(k,clusters[k]) )
    save_dict_on_file(clusters,'betweennes_clustering.pkl')
    
    