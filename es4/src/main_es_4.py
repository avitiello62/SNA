import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('../../')
from es3.src.main_es_3 import isTruthful


def coefficient_based_dataset(number_of_iterations):
    """
    This function creates a dataset assigning for each restaurant a number of stars considering the weighted average of
    food, service and value with respect to the assignment of three random coefficients. To avoid the discrimination for
    missing features, if a restaurant is lack of a feature the algorithm assigns a random value
    :param number_of_iterations:
    :return: restaurant_features, restaurant_stars
    """
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, number_of_iterations):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    restaurant_features.append([food, service, value])
                    if service == -1:
                        sv = random.randint(0, 6)
                    else:
                        sv = service
                    if value == -1:
                        vv = random.randint(0, 6)
                    else:
                        vv = value

                    food_coefficient = random.randint(1, 5)
                    service_coefficient = random.randint(1, 5)
                    value_coefficient = random.randint(1, 5)

                    star_coefficient = (food * food_coefficient + sv * service_coefficient + vv * value_coefficient) / (
                            food_coefficient + service_coefficient + value_coefficient)

                    if star_coefficient >= 3.5:
                        star = 3
                    if 1.7 <= star_coefficient < 3.5:
                        star = 2
                    if star_coefficient < 1.7:
                        star = 1
                    restaurant_stars.append(star)

    return restaurant_features, restaurant_stars


def max_based_dataset(number_of_iterations):
    """
    This function creates a dataset assigning for each restaurant a number of stars considering the max value among its
    food, service and value with respect to a random probability
    :param number_of_iterations:
    :return: restaurant_features,restaurant_stars
    """
    restaurant_features = []
    restaurant_stars = []

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
                    restaurant_features.append(tuple([food, service, value]))
                    restaurant_stars.append(star)

    return restaurant_features, restaurant_stars


def average_based_dataset(number_of_iterations):
    """
    This function creates a dataset assigning for each restaurant a number of stars considering the average value among
    its food, service and value with respect to a random probability
    :param number_of_iterations:
    :return: restaurant_features,restaurant_stars
    """
    restaurant_features = []
    restaurant_stars = []
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
                    restaurant_features.append(tuple([food, service, value]))
                    restaurant_stars.append(star)

    return restaurant_features, restaurant_stars


def totally_random_dataset(number_of_iterations):
    """
    This function creates a random dataset.
    :param number_of_iterations:
    :return: restaurant_features,restaurant_stars
    """
    restaurant_features = []
    restaurant_stars = []

    for i in range(0, number_of_iterations):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    star = random.randint(1, 3)
                    restaurant_features.append(tuple([food, service, value]))
                    restaurant_stars.append(star)

    return restaurant_features, restaurant_stars


def logistic_regression(restaurant_features, restaurant_stars, restaurant_features_test, restaurant_stars_test):
    """

    :param restaurant_features:
    :param restaurant_stars:
    :param restaurant_features_test:
    :param restaurant_stars_test:
    :return:
    """
    log_reg = LogisticRegression()
    log_reg.fit(restaurant_features, restaurant_stars)
    results = {}
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                key = (food, service, value)
                results[key] = log_reg.predict([key])
    accuracy = 0
    for i in range(len(restaurant_features_test)):
        if log_reg.predict([restaurant_features_test[i]]) == restaurant_stars_test[i]:
            accuracy += 1

    return (isTruthful(results)), accuracy / (len(restaurant_stars_test))


def linear_regressor(restaurant_features, restaurant_stars, restaurant_features_test, restaurant_stars_test):
    """

    :param restaurant_features:
    :param restaurant_stars:
    :param restaurant_features_test:
    :param restaurant_stars_test:
    :return:
    """
    lin_reg = LinearRegression(positive=True)
    lin_reg.fit(restaurant_features, restaurant_stars)
    results = {}
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                key = (food, service, value)
                results[key] = np.round(lin_reg.predict([key]))
    accuracy = 0
    for i in range(len(restaurant_features_test)):
        if np.round(lin_reg.predict([restaurant_features_test[i]])) == restaurant_stars_test[i]:
            accuracy += 1

    return isTruthful(results), accuracy / (len(restaurant_stars_test))


def ic_logisticreg(restaurant_features, restaurant_stars, restaurant_features_test, restaurant_stars_test):
    """

    :param restaurant_features:
    :param restaurant_stars:
    :param restaurant_features_test:
    :param restaurant_stars_test:
    :return:
    """
    log_reg = LogisticRegression()
    log_reg.fit(restaurant_features, restaurant_stars)

    # forcing dei parametri
    for i in range(3):
        if log_reg.intercept_[i] < 0:
            log_reg.intercept_[i] = 0

    for i in range(3):
        for j in range(3):
            if log_reg.coef_[i][j] < 0:
                log_reg.coef_[i][j] = 0

    results = {}
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                key = (food, service, value)
                results[key] = log_reg.predict([key])
    accuracy = 0
    for i in range(len(restaurant_features_test)):
        if log_reg.predict([restaurant_features_test[i]]) == restaurant_stars_test[i]:
            accuracy += 1

    return (isTruthful(results)), accuracy / (len(restaurant_stars_test))


def func():
    th_log1 = 0
    acc_log1 = 0
    th_ic1 = 0
    acc_ic1 = 0
    th_lin1 = 0
    acc_lin1 = 0

    th_log2 = 0
    acc_log2 = 0
    th_ic2 = 0
    acc_ic2 = 0
    th_lin2 = 0
    acc_lin2 = 0

    th_log3 = 0
    acc_log3 = 0
    th_ic3 = 0
    acc_ic3 = 0
    th_lin3 = 0
    acc_lin3 = 0

    th_log4 = 0
    acc_log4 = 0
    th_ic4 = 0
    acc_ic4 = 0
    th_lin4 = 0
    acc_lin4 = 0

    dim_test = 1

    X, Y = coefficient_based_dataset(10000)
    for i in range(dim_test):
        Xtest, Ytest = coefficient_based_dataset(1000)
        th, acc = logistic_regression(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_log1 += 1
        acc_log1 += acc

        th, acc = ic_logisticreg(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_ic1 += 1

        acc_ic1 += acc
        th, acc = linear_regressor(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_lin1 += 1
        acc_lin1 += acc
        print("---------------------")

        Xtest, Ytest = max_based_dataset(1000)

        th, acc = logistic_regression(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_log2 += 1
        acc_log2 += acc

        th, acc = ic_logisticreg(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_ic2 += 1
        acc_ic2 += acc
        th, acc = linear_regressor(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_lin2 += 1
        acc_lin2 += acc
        print("---------------------")

        Xtest, Ytest = average_based_dataset(1000)
        th, acc = logistic_regression(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_log3 += 1
        acc_log3 += acc

        th, acc = ic_logisticreg(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_ic3 += 1

        acc_ic3 += acc
        th, acc = linear_regressor(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_lin3 += 1
        acc_lin3 += acc
        print("---------------------")

        Xtest, Ytest = totally_random_dataset(1000)
        th, acc = logistic_regression(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_log4 += 1
        acc_log4 += acc

        th, acc = ic_logisticreg(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_ic4 += 1
        acc_ic4 += acc
        th, acc = linear_regressor(X, Y, Xtest, Ytest)
        print(th, acc)
        if th is True:
            th_lin4 += 1
        acc_lin4 += acc
        print("---------------------")

    print(th_log1 / dim_test, acc_log1 / dim_test)
    print(th_ic1 / dim_test, acc_ic1 / dim_test)
    print(th_lin1 / dim_test, acc_lin1 / dim_test)

    print(th_log2 / dim_test, acc_log2 / dim_test)
    print(th_ic2 / dim_test, acc_ic2 / dim_test)
    print(th_lin2 / dim_test, acc_lin2 / dim_test)

    print(th_log3 / dim_test, acc_log3 / dim_test)
    print(th_ic3 / dim_test, acc_ic3 / dim_test)
    print(th_lin3 / dim_test, acc_lin3 / dim_test)

    print(th_log4 / dim_test, acc_log4 / dim_test)
    print(th_ic4 / dim_test, acc_ic4 / dim_test)
    print(th_lin4 / dim_test, acc_lin4 / dim_test)


if __name__ == '__main__':
    func()
