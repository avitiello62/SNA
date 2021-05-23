import pandas as pd


def read_list_from_csv(csv_name):
    df = pd.read_csv(csv_name)
    top_500_list = df.values.reshape((-1)).tolist()
    return top_500_list


if __name__ == '__main__':
    pagerank_naive = read_list_from_csv("../es2/pagerank_naive.csv")
    pagerank_vectorized = read_list_from_csv("../es2/pagerank_vectorized.csv")
    pagerank_networkx = read_list_from_csv("../es2/pagerank_networkx.csv")

    '''
    1) comparare diverse implementazioni della stessa misura di centralità rispetto alla precisione e alla velocità, per
    precisione intendiamo rispetto al risultato della implementazione naive (oppure la implementazione più precisa se 
    non si può runnare la naive in tempi ragionevoli)
    
    2) valutare similitudini e differenze tra diverse misure di centralità:
        - ci sono nodi che hanno un alto livello di centralità rispetto tutte le misure di centralità?
        - si trovano sempre nella stessa posizione?
        - quali sono le misure di centralità che hanno gli output più simili? 
    '''

    pagerank_naive = set(pagerank_naive)
    pagerank_vectorized = set(pagerank_vectorized)
    pagerank_networkx = set(pagerank_networkx)

    first = pagerank_naive.intersection(pagerank_vectorized)
    second = pagerank_naive.intersection(pagerank_networkx)
    third = pagerank_vectorized.intersection(pagerank_networkx)

    print(len(first))
    print(first)
    print(len(second))
    print(second)
    print(len(third))
    print(third)