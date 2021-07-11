import numpy as np
import random
import networkx as nx


# creazione primo tipo di dataset
def create_ds1():
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, 10000):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    if service == -1:
                        sv = random.randint(0, 5)
                    else:
                        sv = service
                    if value == -1:
                        vv = random.randint(0, 5)
                    else:
                        vv = value
                    food_coefficient = random.randint(1, 5)
                    service_coefficient = random.randint(1, 4)
                    value_coefficient = random.randint(1, 4)
                    star = round((food * food_coefficient + sv * service_coefficient + vv * value_coefficient) / (random.randint(18, 24)))
                    restaurant_features.append([food, service, value])
                    if star >= 3:
                        restaurant_stars.append(3)
                    else:
                        restaurant_stars.append(star + 1)
    return restaurant_features, restaurant_stars


# creazione secondo tipo di dataset
def create_ds2():
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, 10000):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    max = np.array([food, service, value]).max()
                    if max >= 4:
                        prob = random.random()
                        if prob > 0.15 * max:
                            star = 3
                        else:
                            star = 2
                    elif max == 3:
                        prob = random.random()
                        if prob > 0.8:
                            star = 3
                        if 0.3 < prob <= 0.8:
                            star = 2
                        if prob <= 0.3:
                            star = 1
                    else:
                        prob = random.random()
                        if prob > 0.7:
                            star = 2
                        else:
                            star = 1
                    restaurant_features.append([food, service, value])
                    restaurant_stars.append(star)
    return restaurant_features, restaurant_stars


# creazione terzo tipo di dataset
def create_ds3():
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, 10000):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    restaurant_features.append([food, service, value])
                    if service != -1 and value != -1:
                        avg = (food + service + value) / 3
                    elif service == -1:
                        avg = (food + value) / 2
                    elif value == -1:
                        avg = (food + service) / 2
                    elif service == -1 and value == -1:
                        avg = food
                    if avg >= 3.5:
                        star = 3 - random.randint(0, 1)
                    if 1.7 <= avg < 3.5:
                        star = 2 + random.randint(-1, 1)
                    if avg < 1.7:
                        star = 1 + random.randint(0, 1)
                    restaurant_stars.append(star)
    return restaurant_features, restaurant_stars


# creazione quarto dataset
def create_ds4():
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, 10000):
        for f in range(0, 6):
            for s in range(-1, 6):
                for v in range(-1, 6):
                    restaurant_features.append([f, s, v])
                    restaurant_stars.append(random.randint(1, 3))
    return restaurant_features, restaurant_stars


# dato il dataset restituisce un dizionario con la probabilità che una data combinzione di voti ottenga un certo numero di stelle
def prob(restaurant_features, restaurant_stars):
    percentages_map = {}  # mappa delle percentuali
    total_votes_map = {}  # mappa delle combinzioni di voti totali
    for i in range(len(restaurant_features)):
        if tuple(restaurant_features[i]) not in total_votes_map:
            total_votes_map[tuple(restaurant_features[i])] = 1
        else:
            total_votes_map[tuple(restaurant_features[i])] += 1

        if tuple(restaurant_features[i]) + (restaurant_stars[i],) not in percentages_map:
            percentages_map[tuple(restaurant_features[i]) + (restaurant_stars[i],)] = 1
        else:
            percentages_map[tuple(restaurant_features[i]) + (restaurant_stars[i],)] += 1

    # calcolo delle percentuali
    for k in percentages_map:
        percentages_map[k] = percentages_map[k] / total_votes_map[k[:-1]]

    return percentages_map


# doppio mincut in cascata
def mincut1_23(probability_dict):
    # creazione del primo grafo
    G = nx.DiGraph()
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                k = (food, service, value)
                ks = (food, service, value, 1)
                if ks not in probability_dict:
                    probability_dict[ks] = 0
                G.add_edge("s", k,
                           capacity=probability_dict[ks])  # edge con peso uguale alla probabilità che venga asseganta una stella
                if (k[1] == -1 and k[2] == -1):
                    G.add_edge(k, "t", capacity=1 - probability_dict[
                        ks])  # edge con peso uguale alla probabilità che evnga asseganta più di una stella
                else:
                    # combinazione con feature nascoste
                    v1 = (k[0], k[1], -1)
                    v2 = (k[0], -1, k[2])
                    v3 = (k[0], -1, -1)
                    # edge con pesi infiniti
                    if k != v1:
                        G.add_edge(k, v1, capacity=np.inf)
                    if k != v2:
                        G.add_edge(k, v2, capacity=np.inf)
                    if k != v3:
                        G.add_edge(k, v3, capacity=np.inf)
                    G.add_edge(k, "t", capacity=1 - probability_dict[
                        ks])  # edge con peso uguale alla probabilità che venga asseganta più di una stella
    cut_value, partition = nx.minimum_cut(G, "s", "t")
    # costruzione secondo grafo
    G23 = nx.DiGraph()

    for k in partition[1]:
        if k == 't':
            continue
        k2s = k + (2,)
        k3s = k + (3,)
        if k2s not in probability_dict:
            probability_dict[k2s] = 0
        if k3s not in probability_dict:
            probability_dict[k3s] = 0
        p2 = probability_dict[k2s] / (probability_dict[k3s] + probability_dict[k2s])  # probabilità normalizzata
        G23.add_edge("s", k, capacity=p2)  # edge con peso uguale alla probabilità che vengano assegante due stelle
        if (k[1] == -1 and k[2] == -1):
            G23.add_edge(k, "t",
                         capacity=1 - p2)  # edge con peso uguale alla probabilità che vengano assegante più di due stelle
        else:
            # edge con pesi infiniti
            v1 = (k[0], k[1], -1)
            v2 = (k[0], -1, k[2])
            v3 = (k[0], -1, -1)
            if k != v1 and v1 in partition[1]:
                G23.add_edge(k, v1, capacity=np.inf)
            if k != v2 and v2 in partition[1]:
                G23.add_edge(k, v2, capacity=np.inf)
            if k != v3 and v3 in partition[1]:
                G23.add_edge(k, v3, capacity=np.inf)
            G23.add_edge(k, "t",
                         capacity=1 - p2)  # edge con peso uguale alla probabilità che vengano assegante più di due stelle
    cut_value, partition2 = nx.minimum_cut(G23, "s", "t")
    # costruzione del dizionario dei risualtati finali
    mapresults = {}
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                tr = (food, service, value)
                if tr in partition[0]:
                    mapresults[tr] = 1
                    continue
                if tr in partition2[0]:
                    mapresults[tr] = 2
                    continue
                mapresults[tr] = 3
    return mapresults


# permette di verificare che il classificatore rispetti le regole date
def isTrth(m):
    for k in m:
        # controllo rispetto ad un comabinazione in cui sono nascoste due feature
        if k[1] == -1 and k[2] == -1:
            for i in range(0, 6):
                for j in range(0, 6):
                    if m[k] > m[(k[0], i, j)]:
                        # print([k],[k[0],i,j])
                        return False
        # controllo rispetto ad un comabinazione in cui è nascosta solo la feature s
        if k[1] == -1:
            for j in range(0, 6):
                if m[k] > m[(k[0], j, k[2])]:
                    # print([k],[k[0],j,k[2]])
                    return False
        # controllo rispetto ad un comabinazione in cui è nascosta solo la feature v
        if k[2] == -1:
            for j in range(0, 6):
                if m[k] > m[(k[0], k[1], j)]:
                    # print([k],[(k[0],k[1],j)])
                    return False
    return True


# test sui vari datatset
if __name__ == '__main__':
    for i in range(10):
        restaurant_features, restaurant_stars = create_ds1()
        print(isTrth(mincut1_23(prob(restaurant_features, restaurant_stars))))

        restaurant_features, restaurant_stars = create_ds2()
        print(isTrth(mincut1_23(prob(restaurant_features, restaurant_stars))))

        restaurant_features, restaurant_stars = create_ds3()
        print(isTrth(mincut1_23(prob(restaurant_features, restaurant_stars))))

        restaurant_features, restaurant_stars = create_ds4()
        print(isTrth(mincut1_23(prob(restaurant_features, restaurant_stars))))
