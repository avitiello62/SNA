import time
from utils.priorityq import *
from es2.src.hits import *
from es2.src.vect_pagerank import *
import pandas as pd



def top_new(G, measure, k):
    pq = PriorityQueue()
    cen = measure(G)
    print(cen)
    for u in G.nodes():
        pq.add(u, -cen[u])
        # We use negative value because PriorityQueue returns first values whose priority value is lower
    out = []
    for i in range(k):
        out.append(pq.pop())
    return out


def save_list_on_file(top_500_list, name, column_name="TOP 500"):
    df = pd.DataFrame(top_500_list, columns=[column_name])
    df.to_csv(name, index=False)


if __name__ == '__main__':
    G = nx.Graph()
    G = load_dataset("../facebook_large/musae_facebook_edges.csv")

    top_number = 500

    '''print("Pagerank")
    start = time.time()
    top_500_list = top_new(G, pagerank_naive, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        # print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'pagerank_naive.csv')
    print("TIME: ", stop - start)'''

    '''print("Pagerank Vectorized")
    start = time.time()
    top_500_list = top_new(G, vectorized_pagerank, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        #print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'pagerank_vectorized.csv')
    print("TIME: ", stop-start)'''

    '''print("Pagerank di Networkx")
    top_500_list = top_new(G, pagerank, top_number)
    i = 1
    for k in top_500_list:
        # print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'pagerank_networkx.csv')'''

    '''print("Closeness")
    start = time.time()
    top_500_list = top_new(G, closeness, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        # print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'closeness.csv')
    print("TIME: ", stop-start)'''

    '''print("Parallel Closeness")
    start = time.time()
    top_500_list = top_new(G, parallel_closeness, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        # print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'parallel_closeness.csv')
    print("TIME: ", stop - start)'''

    '''print("Degree")
    start = time.time()
    top_500_list = top_new(G, degree, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        #print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'degree_naive.csv')
    print("TIME: ", stop-start)'''

    '''print("Hits Naive")
    start = time.time()
    top_500_list = top_new(G, hits, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        # print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'hits_naive.csv')
    print("TIME: ", stop - start)'''

    '''print("Hits Vectorized")
    start = time.time()
    top_500_list = top_new(G, my_hits, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        # print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'hits_vectorized.csv')
    print("TIME: ", stop - start)'''
