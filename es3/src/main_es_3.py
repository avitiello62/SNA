import numpy as np
import random
import networkx as nx
import time


def max_based_dataset(number_of_iterations):
    """
    This function creates a dataset assigning for each restaurant a number of star considering the max value among its
    food, service and value with respect to a random probability
    :param number_of_iterations:
    :return: dataset
    """
    dataset = {}

    for i in range(0, number_of_iterations):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    v = np.array([food, service, value])
                    max_value = v[np.argmax(v)]
                    probability = random.random()
                    if max_value > 4.5:
                        if probability < 0.1:
                            star = 1
                        elif probability < 0.5:
                            star = 2
                        else:
                            star = 3
                    elif max_value > 3.5:
                        if probability < 0.20:
                            star = 1
                        elif probability < 0.8:
                            star = 2
                        else:
                            star = 3
                    elif max_value > 2.5:
                        if probability < 0.3:
                            star = 1
                        elif probability < 0.95:
                            star = 2
                        else:
                            star = 3
                    else:
                        if probability < 0.15:
                            star = 2
                        else:
                            star = 1
                    dataset[tuple([food, service, value])] = star

    return dataset


def average_based_dataset(number_of_iterations):
    """
    This function creates a dataset assigning for each restaurant a number of star considering the average value among
    its food, service and value with respect to a random probability
    :param number_of_iterations:
    :return: dataset
    """
    dataset = {}
    for i in range(0, number_of_iterations):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):

                    if service != -1 and value != -1:
                        average = (food + service + value) / 3
                    elif service == -1 and value == -1:
                        average = food
                    elif service == -1:
                        average = (food + value) / 2
                    elif value == -1:
                        average = (food + service) / 2

                    if average >= 3.5:
                        star = 3 - random.randint(0, 1)
                    if 1.7 <= average < 3.5:
                        star = 2 + random.randint(-1, 1)
                    if average < 1.7:
                        star = 1 + random.randint(0, 1)
                    dataset[tuple([food, service, value])] = star

    return dataset


def totally_random_dataset(number_of_iterations):
    """
        This function creates a random dataset
        :param number_of_iterations:
        :return: dataset
        """
    dataset = {}

    for i in range(0, number_of_iterations):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    star = random.randint(1, 3)
                    dataset[tuple([food, service, value])] = star
    return dataset


def probability_computation(dataset):
    """
    This function calculates the a priori probability of each tuple (food, service, value) to receive a given stars
    number based on statistical average
    (calculated as: number of (tuple (food, service, value), stars)/number of (tuple (food, service, value), *)
    :param dataset
    :return: percentages_map
    """
    percentages_map = {}  # mappa delle percentuali
    total_votes_map = {}  # mappa delle combinzioni di voti totali
    for key in dataset.keys():
        if key not in total_votes_map:
            total_votes_map[key] = 1
        else:
            total_votes_map[key] += 1

        if key + (dataset[key],) not in percentages_map:
            percentages_map[key + (dataset[key],)] = 1
        else:
            percentages_map[key + (dataset[key],)] += 1
    for k in percentages_map:
        percentages_map[k] = percentages_map[k] / total_votes_map[k[:-1]]

    return percentages_map


def mincut_algorithm(probability_dict):
    """
    This function computes the MinCut algorithm two times in order to create two cut on the graph and classify three
    different classes of restaurants(1 star, 2 stars, 3 stars).
    :param probability_dict:
    :return: mincut_result
    """
    G = nx.DiGraph()
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                k = (food, service, value)
                ks = (food, service, value, 1)
                # If ks is not in probability_dict there is a high probability that ks is a restaurant with a high value
                # in its tuple (food, service, value), so its assigned to the second cut
                if ks not in probability_dict:
                    probability_dict[ks] = 0
                # Creation of the edge linked to "s" with the probability to receive only one star
                G.add_edge("s", k, capacity=probability_dict[ks])
                # If there is at least one feature between service and value, we have to create the edges related to
                # hidden feature
                if k[1] != -1 or k[2] != -1:
                    hidden_feature_service = (k[0], -1, k[2])
                    hidden_feature_value = (k[0], k[1], -1)
                    hidden_both_features = (k[0], -1, -1)
                    #
                    if k != hidden_feature_service:
                        G.add_edge(k, hidden_feature_service, capacity=np.inf)
                    if k != hidden_feature_value:
                        G.add_edge(k, hidden_feature_value, capacity=np.inf)
                    if k != hidden_both_features:
                        G.add_edge(k, hidden_both_features, capacity=np.inf)
                # Creation of the edge linked to "s" with the probability to receive more than one star
                G.add_edge(k, "t", capacity=1 - probability_dict[ks])
    # Computation of the first MinCut algorithm
    cut_value, partition = nx.minimum_cut(G, "s", "t")

    G2 = nx.DiGraph()
    for k in partition[1]:
        if k == 't':
            continue
        k2s = k + (2,)
        k3s = k + (3,)
        # If k2s is not in probability_dict is assigned to the two stars partition
        if k2s not in probability_dict:
            probability_dict[k2s] = 1
        # If k3s is not in probability_dict is assigned to the three stars partition
        if k3s not in probability_dict:
            probability_dict[k3s] = 0

        if probability_dict[k2s] > probability_dict[k3s]:
            probability_cut_2s = probability_dict[k2s]
            probability_cut_3s = 1 - probability_dict[k2s]
        else:
            probability_cut_2s = 1 - probability_dict[k3s]
            probability_cut_3s = probability_dict[k3s]

        # Creation of the edge linked to "s" with the probability to receive two stars
        G2.add_edge("s", k, capacity=probability_cut_2s)

        if k[1] != -1 or k[2] != -1:
            hidden_feature_service = (k[0], -1, k[2])
            hidden_feature_value = (k[0], k[1], -1)
            hidden_both_features = (k[0], -1, -1)

            if k != hidden_feature_service and hidden_feature_service in partition[1]:
                G2.add_edge(k, hidden_feature_service, capacity=np.inf)
            if k != hidden_feature_value and hidden_feature_value in partition[1]:
                G2.add_edge(k, hidden_feature_value, capacity=np.inf)
            if k != hidden_both_features and hidden_both_features in partition[1]:
                G2.add_edge(k, hidden_both_features, capacity=np.inf)

        # Creation of the edge linked to "s" with the probability to receive three stars
        G2.add_edge(k, "t", capacity=probability_cut_3s)

    # Computation of the second MinCut algorithm
    cut_value, partition2 = nx.minimum_cut(G2, "s", "t")

    # Creation of the dict containing the results of the classification
    mincut_result = {}
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                tr = (food, service, value)
                if tr in partition[0]:
                    mincut_result[tr] = 1
                    continue
                if tr in partition2[0]:
                    mincut_result[tr] = 2
                    continue
                mincut_result[tr] = 3
    return mincut_result


def isTruthful(result):
    """
    This function checks the results of the MinCut algorithm and returns True if the result is Truthful with respect of
    the rules, False otherwise
    :param result:
    :return: boolean
    """
    for k in result:
        # Checks the rule when both service and value are missing
        if k[1] == -1 and k[2] == -1:
            for service in range(0, 6):
                for value in range(0, 6):
                    if result[k] > result[(k[0], service, value)]:
                        return False
        # Checks the rule when service is missing
        if k[1] == -1:
            for service in range(0, 6):
                if result[k] > result[(k[0], service, k[2])]:
                    return False
        # Checks the rule when value is missing
        if k[2] == -1:
            for value in range(0, 6):
                if result[k] > result[(k[0], k[1], value)]:
                    return False
    return True


if __name__ == '__main__':
    number_of_iterations = 1000
    datasets = [max_based_dataset(number_of_iterations),
                average_based_dataset(number_of_iterations),
                totally_random_dataset(number_of_iterations)]

    time1 = time.time()
    mincut_dataset_1 = mincut_algorithm(probability_computation(datasets[0]))

    time2 = time.time()
    mincut_dataset_2 = mincut_algorithm(probability_computation(datasets[1]))

    time3 = time.time()
    mincut_dataset_3 = mincut_algorithm(probability_computation(datasets[2]))

    time4 = time.time()

    print(isTruthful(mincut_dataset_1), time2 - time1)
    print(isTruthful(mincut_dataset_1), time3 - time2)
    print(isTruthful(mincut_dataset_1), time4 - time3)
