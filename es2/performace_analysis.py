import pandas as pd


def read_list_from_csv(csv_name):
    df = pd.read_csv(csv_name)
    top_500_list = df.values.reshape((-1)).tolist()
    return top_500_list


if __name__ == '__main__':
    '''
        1) comparare diverse implementazioni della stessa misura di centralità rispetto alla precisione e alla velocità,
        per precisione intendiamo rispetto al risultato della implementazione naive (oppure la implementazione più 
        precisa se non si può runnare la naive in tempi ragionevoli)

        2) valutare similitudini e differenze tra diverse misure di centralità:
            - ci sono nodi che hanno un alto livello di centralità rispetto tutte le misure di centralità?
            - si trovano sempre nella stessa posizione?
            - quali sono le misure di centralità che hanno gli output più simili? 
    '''

    # CSVs Reading
    # Pagerank
    pagerank_naive = read_list_from_csv("../es2/pagerank_naive.csv")
    pagerank_vectorized = read_list_from_csv("../es2/pagerank_vectorized.csv")
    pagerank_networkx = read_list_from_csv("../es2/pagerank_networkx.csv")
    # Closeness
    closeness_naive = read_list_from_csv("../es2/closeness.csv")
    closeness_parallel = read_list_from_csv("../es2/parallel_closeness.csv")
    # HITS

    # Degree

    # Betweenness

    # List to set conversions
    # Pagerank
    pagerank_naive = set(pagerank_naive)
    pagerank_vectorized = set(pagerank_vectorized)
    pagerank_networkx = set(pagerank_networkx)
    # Closeness
    closeness_naive = set(closeness_naive)
    closeness_parallel = set(closeness_parallel)
    # HITS

    # Degree

    # Betweenness

    # Comparations
    # Pagerank vs Pagerank
    pagerank_naive_vs_vectorized = pagerank_naive.intersection(pagerank_vectorized)
    pagerank_naive_vs_networkx = pagerank_naive.intersection(pagerank_networkx)
    pagerank_vectorized_vs_netowrkx = pagerank_vectorized.intersection(pagerank_networkx)

    # Closeness vs Closeness
    closeness_naive_vs_parallel = closeness_naive.intersection(closeness_parallel)

    # Pagerank vs Closeness
    closeness_naive_vs_pagerank_naive = closeness_naive.intersection(pagerank_naive)
    closeness_naive_vs_pagerank_networkx = closeness_naive.intersection(pagerank_naive)
    closeness_naive_vs_pagerank_vectorized = closeness_naive.intersection(pagerank_vectorized)

    closeness_parallel_vs_pagerank_naive = closeness_parallel.intersection(pagerank_naive)
    closeness_parallel_vs_pagerank_networkx = closeness_parallel.intersection(pagerank_naive)
    closeness_parallel_vs_pagerank_vectorized = closeness_parallel.intersection(pagerank_vectorized)

    '''vedere altre comparazioni'''

    # Prints
    print("____________________________________________________________________________________________________")
    print("____________________________________________________________________________________________________")
    print("PERFORMANCE ANALYSIS")
    print("____________________________________________________________________________________________________")
    print("____________________________________________________________________________________________________")
    print("PAGERANK IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("naive vs networkx\nnumber of common top 500 nodes: {}\n".format(len(pagerank_naive_vs_networkx)))
    print("naive vs vectorized\nnumber of common top 500 nodes: {}\n".format(len(pagerank_naive_vs_vectorized)))
    print("vectorized vs networkx\nnumber of common top 500 nodes: {}\n".format(len(pagerank_vectorized_vs_netowrkx)))

    print("____________________________________________________________________________________________________")
    print("CLOSENESS IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("naive vs parallel\nnumber of common top 500 nodes: {}\n".format(len(closeness_naive_vs_parallel)))

    print("____________________________________________________________________________________________________")
    print("HITS IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("something...")

    print("____________________________________________________________________________________________________")
    print("DEGREE IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("something...")

    print("____________________________________________________________________________________________________")
    print("CLOSENESS IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("something...")

    print("____________________________________________________________________________________________________")
    print("CROSS COMPARATIONS")
    print("____________________________________________________________________________________________________")
    print("PAGERANK VS CLOSENESS")
    print("____________________________________________________________________________________________________")

    print("pagerank naive vs closeness parallel\nnumber of common top 500 nodes: {}\n"
          .format(len(closeness_parallel_vs_pagerank_naive)))
    print("pagerank networkx vs closeness parallel\nnumber of common top 500 nodes: {}\n"
          .format(len(closeness_parallel_vs_pagerank_networkx)))
    print("pagerank vectorized vs closeness parallel\nnumber of common top 500 nodes: {}\n"
          .format(len(closeness_parallel_vs_pagerank_vectorized)))
