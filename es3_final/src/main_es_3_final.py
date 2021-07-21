import math
import random
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd

sys.path.append("../../")
from utils.utils import get_random_graph
from es1_final.src.fj_dynamics import FJ_dynamics
from es1_final.src.shapley import f_dist, shapley_closeness_unweighted_graph
from es1.src.betweenness import betweenness, betweenness_parallel

GROUP_NUMBER = 6


def votes_counter(initial_preferences, after_fj_dynamics, after_fj_dynamics_and_manipulation, favourite_candidate):
    """
        This function takes in input the candidate and the preferences of the candidates in three different moments:
            -beginning
            -after fj dynamics
            -after fj dynamics and manipulation
        calculates the votes obtained by the candidates in these three different moments and returns all the counters
        and the votes improvement
        :param initial_preferences:
        :param after_fj_dynamics:
        :param after_fj_dynamics_and_manipulation:
        :param favourite_candidate:
        :param initial_preferences:
        :return: initial_preferences_counter
        :return: after_fj_dynamics_counter
        :return: after_fj_dynamics_and_manipulation_counter
        :return: voter_improvement
        """
    initial_preferences_counter = 0
    after_fj_dynamics_counter = 0
    after_fj_dynamics_and_manipulation_counter = 0

    for pref in initial_preferences.values():
        if pref == favourite_candidate:
            initial_preferences_counter += 1
    for pref in after_fj_dynamics.values():
        if pref == favourite_candidate:
            after_fj_dynamics_counter += 1
    for pref in after_fj_dynamics_and_manipulation.values():
        if pref == favourite_candidate:
            after_fj_dynamics_and_manipulation_counter += 1

    preferences = [initial_preferences_counter, after_fj_dynamics_counter, after_fj_dynamics_and_manipulation_counter]
    return preferences


def plurality_voting(candidates_orientation, nodes_preferences):
    """
        This function takes in input the candidates orientation and the preferences of the nodes in the graphs and
        calculates what is the vote of each node
        :param candidates_orientation:
        :param nodes_preferences:
        :return: preferences
            """
    preferences = {}
    for voter_index, voter in enumerate(nodes_preferences):
        min_dist = np.inf
        for candidate_index, candidate in enumerate(candidates_orientation):
            dist = abs(float(voter) - candidate)
            if dist < min_dist:
                min_dist = dist
                preferences[voter_index] = candidate_index
            elif dist == min_dist:
                if 0.4 <= voter <= 0.6:
                    discrimination = random.randint(0, 1)
                    if discrimination:
                        if candidate > voter:
                            min_dist = dist
                            preferences[voter_index] = candidate_index
                    else:
                        if candidate < voter:
                            min_dist = dist
                            preferences[voter_index] = candidate_index
                if voter > 0.6:
                    if candidate > voter:
                        min_dist = dist
                        preferences[voter_index] = candidate_index
                if voter < 0.4:
                    if candidate < voter:
                        min_dist = dist
                        preferences[voter_index] = candidate_index
    return preferences


def seeds_choice(seed_number, centrality_measure, already_voting_nodes):
    """
        This function takes in input the number of seeds, the closeness of the graph and the nodes that already vote for
        the selected candidate and calculates what are the nodes chosen for manipulation
        :param seed_number:
        :param centrality_measure:
        :param already_voting_nodes:
        :return: seeds
            """
    seeds = []
    while len(seeds) < seed_number and len(centrality_measure) > 0:
        seed = max(centrality_measure, key=centrality_measure.get)
        if seed not in already_voting_nodes:
            seeds.append(seed)
            centrality_measure.pop(seed)
    return seeds


def get_already_voting(preferences, candidate):
    """
        This function takes in input the of the nodes in the graph and the selected candidate and calculates what are the
        nodes that already vote for him/her
        :param preferences:
        :param candidate
        :return: already_voting
            """
    already_voting = []
    for node in preferences:
        pref = preferences[node]
        if pref == candidate:
            already_voting.append(node)
    return already_voting


def get_candidate_intervals(candidates_prob):
    """
        This function takes in input an array containing the probabilities related to candidates and calculates voting
        intervals
        :param candidates_prob:
        :return: intervals
            """
    sorted_candidates = sorted(candidates_prob, key=float)
    intervals = []

    if len(sorted_candidates) == 0:
        return intervals
    elif len(sorted_candidates) == 1:
        intervals.append((0, 1))
        return intervals

    prev_value = (float(sorted_candidates[0]) + float(sorted_candidates[1])) / 2
    intervals.append((0, prev_value))
    x = 1

    while x < len(sorted_candidates) - 1:
        next_value = (float(sorted_candidates[x]) + float(sorted_candidates[x + 1])) / 2
        intervals.append((prev_value, next_value))
        prev_value = next_value
        x += 1

    intervals.append((prev_value, 1))
    return intervals


def get_interval(intervals, candidate):
    """
        This function takes in input an array containing the intervals and the selected candidates and returns the
        voting interval.
        intervals
        :param intervals:
        :param candidate
        :return: interval
            """
    for interval in intervals:
        if interval[0] < candidate <= interval[1]:
            return interval
    return intervals[0]


def get_average_orientation(graph, node, preferences):
    """
        This function takes in input the graph, a node and the preferences array and calculates the average orientation
        of the node's neighborhood.
        voting interval.
        intervals
        :param graph:
        :param node:
        :param preferences:
        :return: orientation
            """
    orientation = 0.0
    for neighbor in graph[node]:
        orientation += preferences[str(neighbor)]
    orientation /= len(graph[node])
    return orientation


def candidates_generation(candidates_number):
    """
        This function takes in input the number of candidates to be generated and generates random values for each one
        in a set of uniform intervals
        :param candidates_number:
        :return: candidates_prob
            """
    candidates_prob = []
    first_value = 0
    second_value = 1 / candidates_number
    for i in range(candidates_number):
        candidates_prob.append(random.uniform(first_value, second_value))
        first_value = second_value
        second_value = second_value + 1 / candidates_number
    return candidates_prob


def manipulation(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    """
        This manipulation function uses shapley closeness to choose seeds and tries to shift the average preference of
        neighbors into the candidate interval without losing seed's vote
    :param graph:
    :param candidates_orientation:
    :param candidate:
    :param seed_number:
    :param nodes_preferencies:
    :return: candidates_prob
    """
    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))

    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    clos = shapley_closeness_unweighted_graph(graph, f_dist)
    seeds = seeds_choice(seed_number, clos, already_voting_nodes)

    stub = {}

    intervals = get_candidate_intervals(candidates_orientation)
    cur_interval = get_interval(intervals, candidates_orientation[candidate])

    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            average_neighborhood_orientation = get_average_orientation(graph, str(index), pref)

            if average_neighborhood_orientation < cur_interval[0]:
                manipulation_factor = cur_interval[1] - 0.001
            elif average_neighborhood_orientation > cur_interval[1]:
                manipulation_factor = cur_interval[0] + 0.001
            else:
                manipulation_factor = candidates_orientation[candidate]
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manipulation = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manipulation.values()))
    results = votes_counter(initial_preferences, after_fj_dynamics, after, candidate)

    print(GROUP_NUMBER, results[0], results[2])
    return results


def manipulation_dummy(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    """
        This manipulation function uses shapley closeness to choose seeds and simply assigns the candidate orientation
        to seeds
    :param graph:
    :param candidates_orientation:
    :param candidate:
    :param seed_number:
    :param nodes_preferencies:
    :return: candidates_prob
    """
    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))

    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    clos = shapley_closeness_unweighted_graph(graph, f_dist)
    seeds = seeds_choice(seed_number, clos, already_voting_nodes)

    stub = {}
    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            manipulation_factor = candidates_orientation[candidate]
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manipulation = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manipulation.values()))
    results = votes_counter(initial_preferences, after_fj_dynamics, after, candidate)

    print(GROUP_NUMBER, results[0], results[2])
    return results


def manipulation_with_hard_influence(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    """
        This manipulation function uses shapley closeness to choose seeds and forces seed's neighbors to change their
        orientation by shifting drastically the seed's orientation. This is done to obtain a great number of votes from
        seed's neighbors but risks to lose seed's vote.
        to seeds
    :param graph:
    :param candidates_orientation:
    :param candidate:
    :param seed_number:
    :param nodes_preferencies:
    :return: candidates_prob
        """
    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))

    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    clos = shapley_closeness_unweighted_graph(graph, f_dist)
    seeds = seeds_choice(seed_number, clos, already_voting_nodes)

    stub = {}

    intervals = get_candidate_intervals(candidates_orientation)
    cur_interval = get_interval(intervals, candidates_orientation[candidate])
    cur_interval_average = (cur_interval[0] + cur_interval[1]) / 2

    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            average_neighborhood_orientation = get_average_orientation(graph, str(index), pref)
            manipulation_factor = 2 * cur_interval_average - average_neighborhood_orientation
            if manipulation_factor > 1:
                manipulation_factor = 1
            if manipulation_factor < 0:
                manipulation_factor = 0
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manipulation = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manipulation.values()))
    results = votes_counter(initial_preferences, after_fj_dynamics, after, candidate)

    print(GROUP_NUMBER, results[0], results[2])
    return results


def manipulation_dummy_with_parallel_betweenness(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    """
        This manipulation function uses parallel betweenness to choose seeds and simply assigns the candidate orientation
        to seeds
    :param graph:
    :param candidates_orientation:
    :param candidate:
    :param seed_number:
    :param nodes_preferencies:
    :return: candidates_prob
        """
    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))
    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    _, betw = betweenness_parallel(graph,8)
    seeds = seeds_choice(seed_number, betw, already_voting_nodes)

    stub = {}
    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            manipulation_factor = candidates_orientation[candidate]
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manipulation = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manipulation.values()))
    results = votes_counter(initial_preferences, after_fj_dynamics, after, candidate)

    print(GROUP_NUMBER, results[0], results[2])
    return results


def manipulation_with_hard_influence_with_parallel_betweenness(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    """
        This manipulation function uses parallel betweenness to choose seeds and forces seed's neighbors to change their
        orientation by shifting drastically the seed's orientation. This is done to obtain a great number of votes from
        seed's neighbors but risks to lose seed's vote.
    :param graph:
    :param candidates_orientation:
    :param candidate:
    :param seed_number:
    :param nodes_preferencies:
    :return: candidates_prob
        """
    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference
    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))
    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)
    _, betw = betweenness_parallel(graph,8)
    seeds = seeds_choice(seed_number, betw, already_voting_nodes)

    stub = {}
    intervals = get_candidate_intervals(candidates_orientation)
    cur_interval = get_interval(intervals, candidates_orientation[candidate])
    cur_interval_average = (cur_interval[0] + cur_interval[1]) / 2
    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            average_neighborhood_orientation = get_average_orientation(graph, str(index), pref)
            manipulation_factor = 2 * cur_interval_average - average_neighborhood_orientation
            if manipulation_factor > 1:
                manipulation_factor = 1
            if manipulation_factor < 0:
                manipulation_factor = 0
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manipulation = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manipulation.values()))
    results = votes_counter(initial_preferences, after_fj_dynamics, after, candidate)

    print(GROUP_NUMBER, results[0], results[2])
    return results


def manipulation_with_parallel_betweenness(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    """
        This manipulation function uses parallel betweenness to choose seeds and tries to shift the average preference of
        neighbors into the candidate interval without losing seed's vote
    :param graph:
    :param candidates_orientation:
    :param candidate:
    :param seed_number:
    :param nodes_preferencies:
    :return: candidates_prob
        """
    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference
    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))
    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    _, betw = betweenness_parallel(graph)
    seeds = seeds_choice(seed_number, betw, already_voting_nodes)

    stub = {}
    intervals = get_candidate_intervals(candidates_orientation)
    cur_interval = get_interval(intervals, candidates_orientation[candidate])

    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            average_neighborhood_orientation = get_average_orientation(graph, str(index), pref)

            if average_neighborhood_orientation < cur_interval[0]:
                manipulation_factor = cur_interval[1] - 0.001
            elif average_neighborhood_orientation > cur_interval[1]:
                manipulation_factor = cur_interval[0] + 0.001
            else:
                manipulation_factor = candidates_orientation[candidate]
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manipulation = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manipulation.values()))
    results = votes_counter(initial_preferences, after_fj_dynamics, after, candidate)

    print(GROUP_NUMBER, results[0], results[2])
    return results


def test_function(manipulation_function, graph, candidates_prob, seed_number, nodes_preferences):
    result = {}
    for i in tqdm(range(len(candidates_prob))):
        candidate = i
        votes = {}
        for num in seed_number:
            manipulation = manipulation_function(graph, candidates_prob, candidate, num, nodes_preferences)
            votes[num] = manipulation
        result[i] = votes

    tab = {}

    tab['*'] = ['start', '10', '15', '20', '25', '30', '35', '40', '45', '50']
    for i in range(len(candidates_prob)):
        tab[str(i)] = [result[i][10][1]]
        for j in range(10, 55, 5):
            tab[str(i)].append(result[i][j][2])
    df = pd.DataFrame(data=tab)
    print(df.head())

    return df


if __name__ == '__main__':
    random.seed=11
    # Graph generation
    numNodes = 250
    density = 0.3
    graph = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * density), False)

    # Candidates generation
    candidates_number = 5
    candidates_prob = candidates_generation(candidates_number)

    # Preferences generation
    nodes_pref = []
    for _ in range(graph.number_of_nodes()):
        nodes_pref.append(random.uniform(0, 1))

    # Seed
    seed_number = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Test for each manipulation function with the seed_number array
    df_manipulation = test_function(manipulation, graph, candidates_prob, seed_number, nodes_pref)
    df_manipulation.to_csv("results/manipulation.csv")

    df_hard_influence = test_function(manipulation_with_hard_influence, graph, candidates_prob, seed_number,
                                      nodes_pref)
    df_hard_influence.to_csv("results/manipulation_with_hard_influence.csv")

    df_dummy = test_function(manipulation_dummy, graph, candidates_prob, seed_number, nodes_pref)
    df_dummy.to_csv("results/manipulation_dummy.csv")

    df_betw = test_function(manipulation_with_parallel_betweenness, graph, candidates_prob, seed_number,
                                  nodes_pref)
    df_betw.to_csv("results/manipulation_with_parallel_betweenness.csv")

    df_dummy_betw = test_function(manipulation_dummy_with_parallel_betweenness, graph, candidates_prob, seed_number,
                                     nodes_pref)
    df_dummy_betw.to_csv("results/manipulation_dummy_with_parallel_betweenness.csv")

    df_hard_influence_betw = test_function(manipulation_with_hard_influence_with_parallel_betweenness, graph, candidates_prob, seed_number,
                                     nodes_pref)
    df_hard_influence_betw.to_csv("results/manipulation_with_hard_influence_with_parallel_betweenness.csv")