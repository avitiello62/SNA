import math
import os
import random
import sys
from itertools import permutations

import networkx as nx
import numpy as np
from networkx.algorithms.community import asyn_fluidc, asyn_lpa_communities
from numpy import linspace

from Common.DynamicGraphGeneration import get_random_graph
from Common.FixedGraph import get_fixed_graph, get_fixed_num_graph
from Common.DrawGraph import DrawGraph
from Common.LoadGraph import load_graph
from Common.Top import get_top_nodes
from Final_Exercise1.FJ_Dynamics import FJ_dynamics
from Exercise2.NaiveImplementations import degree, closeness
from Exercise2.Betweenness import betweenness_parallel
from Exercise1.NaiveImplementations import betweenness
from Final_Exercise1.Shapley import shapley_closeness
from Final_Exercise2.NetworkModels import *


def aftermath(prev, middle, after, candidate):
    prev_cnt = 0
    after_cnt = 0
    middle_cnt = 0
    for pref in prev.values():
        if pref == candidate:
            prev_cnt += 1
    for pref in middle.values():
        if pref == candidate:
            middle_cnt += 1
    for pref in after.values():
        if pref == candidate:
            after_cnt += 1
    return prev_cnt, middle_cnt, after_cnt, after_cnt - prev_cnt


def plurality_voting(candidates_prob, nodes_pref):
    preferences = {}
    for voter_index, voter in enumerate(nodes_pref):
        min_dist = 2
        for cand_index, candidate in enumerate(candidates_prob):
            dist = abs(float(voter) - candidate)

            if dist < min_dist:
                min_dist = dist
                preferences[voter_index] = cand_index
            elif dist == min_dist:
                if candidate < voter:
                    min_dist = dist
                    preferences[voter_index] = cand_index
    return preferences


def custom_logistic_func(value, a, c, k):
    return (-c / (1 + a * math.exp(-k * value))) + 2


def get_best_seeds(graph, candidates, pref_candidate, budget, preferences, closeness, betweenness, percent, already_voting):
    nodes_closeness = max(1, (int(budget * percent)))
    seeds = []

    # Interpret the centrality value as a sort of "ideality" of being selected as a seed
    # Multiply such value by the fraction of neighbors which are actually influenceable
    for node in graph.nodes():
        for neighbor in graph[node]:
            if neighbor not in already_voting:
                difference = abs(preferences[int(neighbor)] - candidates[pref_candidate])
                multiplier = custom_logistic_func(difference, 170, 1.5, 14)
                closeness[node] *= multiplier
                betweenness[node] *= multiplier

    while len(seeds) < nodes_closeness and len(closeness) > 0:
        seed = max(closeness, key=closeness.get)
        if seed not in already_voting:
            closeness.pop(seed)
            seeds.append(seed)
    while len(seeds) < budget and len(betweenness) > 0:
        seed = max(betweenness, key=betweenness.get)
        if seed not in already_voting:
            betweenness.pop(seed)
            if seed not in seeds:
                seeds.append(seed)
    return seeds


def get_already_voting(preferences, candidate):
    already_voting = []
    for node in preferences:
        pref = preferences[node]
        if pref == candidate:
            already_voting.append(node)
    return already_voting


def get_candidate_intervals(candidates_prob):
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
    for interval in intervals:
        if interval[0] < candidate <= interval[1]:
            return interval
    return intervals[0]  # In case the candidate is the left-most oriented one


def get_average_orientation(G, node, pref):
    total = 0.0
    for neighbor in G[node]:
        total += pref[str(neighbor)]
    total /= len(G[node])
    return total


def manipulation(G, candidates_prob, candidate, seed_num, nodes_pref):

    # Re-mapping graph nodes to strings (if they are ints)
    mapping = {}
    for node in G.nodes():
        mapping[node] = str(node)
    nx.relabel_nodes(G, mapping, False)  # Re-labelling is done in-place

    prev = plurality_voting(candidates_prob, nodes_pref)
    pref = {}
    stub = {}

    for index, node in enumerate(nodes_pref):
        stub[str(index)] = 0.5

    # Create dict node -> preference
    for index, preference in enumerate(nodes_pref):
        pref[str(index)] = preference

    # Run the dynamics without influence
    mid = FJ_dynamics(G, pref.copy(), stub, num_iter=200)
    middle = plurality_voting(candidates_prob, list(mid.values()))
    already_voting_seeds = get_already_voting(middle, candidate)

    # Select the best seeds
    # Compute centrality measures
    _, betw = betweenness_parallel(G, 4)
    clos = shapley_closeness(G)
    seeds = get_best_seeds(G, candidates_prob,  candidate, seed_num, nodes_pref, clos, betw, 0.85, already_voting_seeds)

    stub = {}
    intervals = get_candidate_intervals(candidates_prob)
    # Create dict node -> stubbornness
    for index, node in enumerate(nodes_pref):
        if str(index) in seeds:
            stub[str(index)] = 1
            seed_value = get_average_orientation(G, str(index), pref)
            cur_interval = get_interval(intervals, candidates_prob[candidate])  # Interval of the preferred candidate
            if seed_value > cur_interval[1]:
                pref[str(index)] = cur_interval[1]
            elif seed_value <= cur_interval[0]:
                pref[str(index)] = cur_interval[0] + 0.001  # Epsilon
            else:
                pref[str(index)] = candidates_prob[candidate]
        else:
            stub[str(index)] = 0.5

    manip = FJ_dynamics(G, pref, stub, num_iter=200)
    after = plurality_voting(candidates_prob, list(manip.values()))
    prev_cnt, middle_cnt, after_cnt, increment = amath = aftermath(prev, middle, after, candidate)
    print("12,", str(prev_cnt) + ",", after_cnt)
    return prev, middle, after, amath





##################################################
# This is code used to perform tests and debug
##################################################

numNodes = 250
density = 0.3
random_graph = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * density), False)


candidates_prob = []
for _ in range(5):
    candidates_prob.append(random.uniform(0, 1))
# Override:
candidates_prob = [0.67, 0.36, 0.57, 0.34, 0.81]
candidate = random.choice(range(len(candidates_prob)))
# Override:
candidate = 4
# c = np.argmax(np.array(p))
print("Candidates: ", candidates_prob)

nodes_pref = []
for _ in range(random_graph.number_of_nodes()):
    nodes_pref.append(random.uniform(0, 1))
print("Random values generated")

increments = {}
max_increment_num_nodes = 0
max_increment = 0

p_voting = plurality_voting(candidates_prob, nodes_pref)

num_of_nodes = [10, 15, 20, 25, 30, 40, 50]

for num in num_of_nodes:

    print("Preferred candidate: " + str(candidate))
    # print("Seeds: " + str(seeds) + "\t\t# of seeds: " + str(len(seeds)))
    (prev, middle, after, amath) = manipulation(random_graph, candidates_prob, candidate, num, nodes_pref)
    prev_cnt, middle_cnt, after_cnt, increment = amath
    print("Previous voting: " + str(prev))
    print("Un-manipulated voting: " + str(middle))
    print("Manipulated voting: " + str(after))
    print("Previously voting: ", prev_cnt, "\t\tWith Dynamics: ", middle_cnt, "\t\tNow voting: ", after_cnt)
    print("Increment without seeds: " + str(increment - num) + "\t\tTotal Increment: " + str(increment))
    print("------------------------------------------------------------------------------------------------\n")

