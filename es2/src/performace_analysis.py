import pandas as pd


def read_list_from_csv(csv_name):
    df = pd.read_csv(csv_name)
    top_500_list = df.values.reshape((-1)).tolist()
    return top_500_list


def positional_comparation(set1, set2):
    if len(set1) == len(set2):
        z = []
        set1 = list(set1)
        set2 = list(set2)

        for x in range(len(set1)):
            # print(set1[x],set2[x])
            if set1[x] == set2[x]:
                z.append(x)
    return z


def performance_analisys():
    normalization_factor = 500
    print("____________________________________________________________________________________________________")
    print("____________________________________________________________________________________________________")
    print("PERFORMANCE ANALYSIS")
    print("____________________________________________________________________________________________________")
    # CSVs Reading
    print("CSVs Reading")
    # Degree
    print("Reading Degree Results")
    degree_naive = read_list_from_csv("../es2__/degree_naive.csv")
    # Closeness
    print("Reading Closeness Results")
    closeness_naive = read_list_from_csv("../es2__/closeness.csv")
    closeness_parallel = read_list_from_csv("../es2__/parallel_closeness.csv")
    # Betweenness
    print("Reading Betweenness Results")
    # Pagerank
    print("Reading Pagerank Results")
    pagerank_naive = read_list_from_csv("../es2__/pagerank_naive.csv")
    pagerank_vectorized = read_list_from_csv("../es2__/pagerank_vectorized.csv")
    pagerank_networkx = read_list_from_csv("../es2__/pagerank_networkx.csv")
    # HITS
    print("Reading HITS Results")
    hits_naive = read_list_from_csv("../es2__/hits_naive.csv")
    hits_parallel = read_list_from_csv("../es2__/hits_vectorized.csv")

    # List to set conversions
    # Pagerank
    pagerank_naive = set(pagerank_naive)
    pagerank_vectorized = set(pagerank_vectorized)
    pagerank_networkx = set(pagerank_networkx)
    # Closeness
    closeness_naive = set(closeness_naive)
    closeness_parallel = set(closeness_parallel)
    # HITS
    hits_naive = set(hits_naive)
    hits_parallel = set(hits_parallel)
    # Degree
    degree_naive = set(degree_naive)
    # Betweenness

    # Same Algorithm Comparations
    print("____________________________________________________________________________________________________")
    print("SAME ALGORITHM COMPARATIONS")
    print("____________________________________________________________________________________________________")
    # Degree vs Degree
    print("____________________________________________________________________________________________________")
    print("DEGREE IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")
    print(
        "\ndegree has been implemented only in naive way because it is very fast and does not need any optimization\n")
    # Closeness vs Closeness
    closeness_naive_vs_parallel_similarity = len(
        closeness_naive.intersection(closeness_parallel)) / normalization_factor
    closeness_naive_vs_parallel_equality = len(
        positional_comparation(closeness_naive, closeness_parallel)) / normalization_factor
    print("____________________________________________________________________________________________________")
    print("CLOSENESS IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("naive vs parallel\n"
          "\tsimilarity rate: {}\n"
          "\tequality rate: {}\n".format(closeness_naive_vs_parallel_similarity, closeness_naive_vs_parallel_equality))
    # Betweenness vs Betweenness
    print("____________________________________________________________________________________________________")
    print("BETWEENNESS IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")
    print(
        "\nno betweenness results\n")
    # Pagerank vs Pagerank
    pagerank_naive_vs_vectorized_similarity = len(
        pagerank_naive.intersection(pagerank_vectorized)) / normalization_factor
    pagerank_naive_vs_networkx_similarity = len(pagerank_naive.intersection(pagerank_networkx)) / normalization_factor
    pagerank_vectorized_vs_netowrkx_similarity = len(
        pagerank_vectorized.intersection(pagerank_networkx)) / normalization_factor

    pagerank_naive_vs_vectorized_equality = len(
        positional_comparation(pagerank_naive, pagerank_vectorized)) / normalization_factor
    pagerank_naive_vs_networkx_equality = len(
        positional_comparation(pagerank_naive, pagerank_networkx)) / normalization_factor
    pagerank_vectorized_vs_netowrkx_equality = len(
        positional_comparation(pagerank_vectorized, pagerank_networkx)) / normalization_factor
    print("____________________________________________________________________________________________________")
    print("PAGERANK IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("naive vs networkx\n"
          "\tsimilarity rate: {}\n"
          "\tequality rate:{}\n".format(pagerank_naive_vs_networkx_similarity, pagerank_naive_vs_networkx_equality))
    print("naive vs vectorized\n"
          "\tsimilarity rate: {}\n"
          "\tequality rate:{}\n".format(pagerank_naive_vs_vectorized_similarity, pagerank_naive_vs_vectorized_equality))
    print("vectorized vs networkx\n"
          "\tsimilarity rate: {}\n"
          "\tequality rate:{}\n".format(pagerank_vectorized_vs_netowrkx_similarity,
                                        pagerank_vectorized_vs_netowrkx_equality))

    # HITS vs HITS
    hits_naive_vs_parallel_similarity = len(hits_naive.intersection(hits_parallel)) / normalization_factor

    hits_naive_vs_parallel_equality = len(positional_comparation(hits_naive, hits_parallel)) / normalization_factor
    print("____________________________________________________________________________________________________")
    print("HITS IMPLEMENTATIONS")
    print("____________________________________________________________________________________________________")

    print("naive vs parallel\n"
          "\tsimilarity rate: {}\n"
          "\tequality rate: {}\n".format(hits_naive_vs_parallel_similarity, hits_naive_vs_parallel_equality))
    return


if __name__ == '__main__':
    performance_analisys()
