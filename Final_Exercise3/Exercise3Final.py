import math
import random
import networkx as nx
import numpy as np
import sys
from tqdm import tqdm
from numpy.lib.function_base import average
import pandas as pd
sys.path.append("../")
from utils.utils import get_random_graph
from es1_final.src.fj_dynamics import FJ_dynamics
from es1_final.src.shapley import shapley_closeness , f_dist, shapley_closeness_unweighted_graph


def votes_counter(initial_preferences, after_fj_dynamics, after_fj_dynamics_and_manipulation, favourite_candidate):
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

    votes_improvement = after_fj_dynamics_and_manipulation_counter - initial_preferences_counter

    return initial_preferences_counter, after_fj_dynamics_counter, after_fj_dynamics_and_manipulation_counter, votes_improvement


def plurality_voting(candidates_orientation, nodes_preferences):
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


def get_best_seeds(graph, candidates_orientation, candidate, seed_number, nodes_preferencies, closeness, already_voting_nodes):
    seeds = []
    
    while len(seeds) < seed_number and len(closeness) > 0:
        seed = max(closeness, key=closeness.get)
        if seed not in already_voting_nodes:
            seeds.append(seed)
            closeness.pop(seed)
    return seeds

def get_best_seeds_version_average(graph, candidates_orientation, candidate, seed_number, nodes_preferencies, closeness, already_voting_nodes,pref):
    seeds = []
    neighbour_distance=[]
    coefficient={}
    for node in graph.nodes():
        
        average=get_average_orientation(graph,node,pref)
        average_diff=abs(average-candidates_orientation[candidate])
        if average_diff>0.50:
            coeff=1
        else:
            coeff=2
        coefficient[node]=closeness[node]*coeff
    

    while len(seeds) < seed_number and len(coefficient) > 0:
        seed = max(coefficient, key=closeness.get)
        if seed not in already_voting_nodes:
            seeds.append(seed)
            coefficient.pop(seed)
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


def manipulation(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    # Re-mapping graph nodes to strings (if they are ints)
    mapping = {}
    for node in graph.nodes():
        mapping[node] = str(node)
    nx.relabel_nodes(graph, mapping, False)  # Re-labelling is done in-place

    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    # Run the dynamics without influence
    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))

    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    # Select the best seeds
    # Compute centrality measures
    clos = shapley_closeness_unweighted_graph(graph,f_dist)
    seeds = get_best_seeds(graph, candidates_orientation, candidate, seed_number, nodes_preferencies, clos, already_voting_nodes)

    stub = {}

    intervals = get_candidate_intervals(candidates_orientation)
    # Create dict node -> stubbornness
    cur_interval = get_interval(intervals, candidates_orientation[candidate])# Interval of the preferred candidate
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

    manip = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manip.values()))
    prev_cnt, middle_cnt, after_cnt, increment = amath = votes_counter(initial_preferences, after_fj_dynamics,
                                                                            after, candidate)
    #print("12,", str(prev_cnt) + ",", after_cnt)
    return initial_preferences, after_fj_dynamics, after, amath

def manipulation_with_vote(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    # Re-mapping graph nodes to strings (if they are ints)
    mapping = {}
    for node in graph.nodes():
        mapping[node] = str(node)
    nx.relabel_nodes(graph, mapping, False)  # Re-labelling is done in-place

    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    # Run the dynamics without influence
    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))

    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    # Select the best seeds
    # Compute centrality measures
    clos = shapley_closeness_unweighted_graph(graph,f_dist)
    seeds = get_best_seeds(graph, candidates_orientation, candidate, seed_number, nodes_preferencies, clos, already_voting_nodes)

    stub = {}

    intervals = get_candidate_intervals(candidates_orientation)
    # Create dict node -> stubbornness
    cur_interval = get_interval(intervals, candidates_orientation[candidate])# Interval of the preferred candidate
    

    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            average_neighborhood_orientation = get_average_orientation(graph, str(index), pref)
            
            if average_neighborhood_orientation < cur_interval[0]:
                manipulation_factor = cur_interval[1]
            elif average_neighborhood_orientation > cur_interval[1]:
                manipulation_factor = cur_interval[0]+0.001
            else :
                 manipulation_factor = candidates_orientation[candidate]
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manip = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manip.values()))
    prev_cnt, middle_cnt, after_cnt, increment = amath = votes_counter(initial_preferences, after_fj_dynamics,
                                                                            after, candidate)
    #print("12,", str(prev_cnt) + ",", after_cnt)
    return initial_preferences, after_fj_dynamics, after, amath





def manipulation_version_average(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    # Re-mapping graph nodes to strings (if they are ints)
    mapping = {}
    for node in graph.nodes():
        mapping[node] = str(node)
    nx.relabel_nodes(graph, mapping, False)  # Re-labelling is done in-place

    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    # Run the dynamics without influence
    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))

    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    # Select the best seeds
    # Compute centrality measures
    clos = shapley_closeness(graph,f_dist)
    seeds = get_best_seeds_version_average(graph, candidates_orientation, candidate, seed_number, nodes_preferencies, clos, already_voting_nodes)

    stub = {}

    intervals = get_candidate_intervals(candidates_orientation)
    # Create dict node -> stubbornness
    cur_interval = get_interval(intervals, candidates_orientation[candidate])# Interval of the preferred candidate
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

    manip = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manip.values()))
    prev_cnt, middle_cnt, after_cnt, increment = amath = votes_counter(initial_preferences, after_fj_dynamics,
                                                                            after, candidate)
    #print("12,", str(prev_cnt) + ",", after_cnt)
    return initial_preferences, after_fj_dynamics, after, amath

def manipulation_with_vote_version_average(graph, candidates_orientation, candidate, seed_number, nodes_preferencies):
    # Re-mapping graph nodes to strings (if they are ints)
    mapping = {}
    for node in graph.nodes():
        mapping[node] = str(node)
    nx.relabel_nodes(graph, mapping, False)  # Re-labelling is done in-place

    initial_preferences = plurality_voting(candidates_orientation, nodes_preferencies)
    pref = {}
    stub = {}

    for index, preference in enumerate(nodes_preferencies):
        stub[str(index)] = 0.5
        pref[str(index)] = preference

    # Run the dynamics without influence
    fj_dynamics_output = FJ_dynamics(graph, pref.copy(), stub, num_iter=200)
    after_fj_dynamics = plurality_voting(candidates_orientation, list(fj_dynamics_output.values()))

    already_voting_nodes = get_already_voting(after_fj_dynamics, candidate)

    # Select the best seeds
    # Compute centrality measures
    clos = shapley_closeness_unweighted_graph(graph,f_dist)
    seeds = get_best_seeds_version_average(graph, candidates_orientation, candidate, seed_number, nodes_preferencies, clos, already_voting_nodes,pref)

    stub = {}

    intervals = get_candidate_intervals(candidates_orientation)
    # Create dict node -> stubbornness
    cur_interval = get_interval(intervals, candidates_orientation[candidate])# Interval of the preferred candidate
    

    for index, node in enumerate(nodes_preferencies):
        if str(index) in seeds:
            stub[str(index)] = 1
            average_neighborhood_orientation = get_average_orientation(graph, str(index), pref)
            
            if average_neighborhood_orientation < cur_interval[0]:
                manipulation_factor = cur_interval[1]
            elif average_neighborhood_orientation > cur_interval[1]:
                manipulation_factor = cur_interval[0]+0.001
            else :
                 manipulation_factor = candidates_orientation[candidate]
            pref[str(index)] = manipulation_factor
        else:
            stub[str(index)] = 0.5

    manip = FJ_dynamics(graph, pref, stub, num_iter=200)
    after = plurality_voting(candidates_orientation, list(manip.values()))
    prev_cnt, middle_cnt, after_cnt, increment = amath = votes_counter(initial_preferences, after_fj_dynamics,
                                                                            after, candidate)
    #print("12,", str(prev_cnt) + ",", after_cnt)
    return initial_preferences, after_fj_dynamics, after, amath

##################################################
# This is code used to perform tests and debug
##################################################

numNodes = 250
density = 0.3
random_graph = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * density), False)

candidates_prob = []

# Override:
candidates_prob = [0.0, 0.25, 0.50, 0.75, 1.0]
#candidate = random.choice(range(len(candidates_prob)))


nodes_pref = []
for _ in range(random_graph.number_of_nodes()):
    nodes_pref.append(random.uniform(0, 1))


increments = {}
max_increment_num_nodes = 0
max_increment = 0
num_of_nodes = [10,15,20,25,30,35,40,45,50]
result1={}

for i in tqdm(range(5)):
    candidate=i
    votes={}
    for num in num_of_nodes:
        (prev, middle, after, amath) = manipulation_with_vote(random_graph, candidates_prob, candidate, num, nodes_pref)
        prev_cnt, middle_cnt, after_cnt, increment = amath
        #print("Previously voting: ", prev_cnt, "\t\tWith Dynamics: ", middle_cnt, "\t\tNow voting: ", after_cnt)
        #print("Increment without seeds: " + str(increment - num) + "\t\tTotal Increment: " + str(increment))
        #print("------------------------------------------------------------------------------------------------\n")
        votes[num]=[prev_cnt,middle_cnt,after_cnt]
    result1[i]=votes




result2={}
for i in tqdm(range(5)):
    candidate=i
    votes={}
    for num in num_of_nodes:
        (prev, middle, after, amath) = manipulation_with_vote_version_average(random_graph, candidates_prob, candidate, num, nodes_pref)
        prev_cnt, middle_cnt, after_cnt, increment = amath
        #print("Previously voting: ", prev_cnt, "\t\tWith Dynamics: ", middle_cnt, "\t\tNow voting: ", after_cnt)
        #print("Increment without seeds: " + str(increment - num) + "\t\tTotal Increment: " + str(increment))
        #print("------------------------------------------------------------------------------------------------\n")
        votes[num]=[prev_cnt,middle_cnt,after_cnt]
    result2[i]=votes

tab={}

tab['*']=['start','10','15','20','25','30','35','40','45','50']
for i in range(5):
    tab[str(i)]=[result1[i][10][1]]
    for j in range(10,55,5):
        tab[str(i)].append(result1[i][j][2])
df=pd.DataFrame(data=tab)
print(df.head())
df.to_csv("dict4.csv")


tab={}

tab['*']=['start','10','15','20','25','30','35','40','45','50']
for i in range(5):
    tab[str(i)]=[result2[i][10][1]]
    for j in range(10,55,5):
        tab[str(i)].append(result2[i][j][2])
df=pd.DataFrame(data=tab)
print(df.head())
df.to_csv("dict5.csv")