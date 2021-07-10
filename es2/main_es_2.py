import time
from es2.src.closeness import *
from es2.src.hits import *
from es2.src.pagerank import *
from utils.lesson2 import *
from utils.priorityq import *
from es2.src.vect_pagerank import *
from networkx.algorithms.link_analysis.pagerank_alg import pagerank, pagerank_numpy


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

    print("Hits Naive")
    start = time.time()
    top_500_list = top_new(G, hits, top_number)
    stop = time.time()
    i = 1
    for k in top_500_list:
        # print("Position {}: node = {}".format(i, k))
        i += 1
    save_list_on_file(top_500_list, 'hits_naive.csv')
    print("TIME: ", stop - start)

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
