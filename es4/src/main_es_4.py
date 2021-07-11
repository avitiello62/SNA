import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import random


# creazione primo dataset
def create_ds1(dim):
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, dim):
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
                    star = round((food * food_coefficient + sv * service_coefficient + vv * value_coefficient) / (
                        random.randint(18, 24)))
                    restaurant_features.append([food, service, value])
                    if star >= 3:
                        restaurant_stars.append(3)
                    else:
                        restaurant_stars.append(star + 1)
    return restaurant_features, restaurant_stars


def create_ds2(dim):
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, dim):
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


# creazione dataset 3
def create_ds3(dim):
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, dim):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    restaurant_features.append([food, service, value])
                    if service != -1 and value != -1:
                        avg = (food + service + value) / 3
                    if service == -1:
                        avg = (food + value) / 2
                    if value == -1:
                        avg = (food + service) / 2
                    if service == -1 and value == -1:
                        avg = food
                    if avg >= 3.5:
                        star = 3 - random.randint(0, 1)
                    if 1.7 <= avg < 3.5:
                        star = 2 + random.randint(-1, 1)
                    if avg < 1.7:
                        star = 1 + random.randint(0, 1)
                    restaurant_stars.append(star)
    return restaurant_features, restaurant_stars


def create_ds4(dim):
    restaurant_features = []
    restaurant_stars = []
    for i in range(0, dim):
        for food in range(0, 6):
            for service in range(-1, 6):
                for value in range(-1, 6):
                    restaurant_features.append([food, service, value])
                    restaurant_stars.append(random.randint(1, 3))
    return restaurant_features, restaurant_stars


# permette di verificare che il classificatore rispetti le regole date
def isTrth(m):
    print(len(m))
    for k in m:
        if k[1] == -1 and k[2] == -1:
            for i in range(0, 6):
                for j in range(0, 6):
                    if m[k] > m[(k[0], i, j)]:
                        print([k], [k[0], i, j])
                        print(m[k], m[k[0], i, j])
                        return False
        if k[1] == -1:
            for j in range(0, 6):
                if m[k] > m[(k[0], j, k[2])]:
                    print([k], [k[0], j, k[2]])
                    print(m[k], m[k[0], j, k[2]])
                    return False
        if k[2] == -1:
            for j in range(0, 6):
                if m[k] > m[(k[0], k[1], j)]:
                    print([k], [(k[0], k[1], j)])
                    print(m[k], m[(k[0], k[1], j)])
                    return False
    return True


# regressore logistico
def logistic_regression(restaurant_features, restaurant_stars, restaurant_features_test, restaurant_stars_test):
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

    return (isTrth(results)), accuracy / (len(restaurant_stars_test))


# regressore lineare con pesi positivi
def linear_regressor(restaurant_features, restaurant_stars, restaurant_features_test, restaurant_stars_test):
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

    return isTrth(results), accuracy / (len(restaurant_stars_test))


# incentive-compatible logistic regression
def ic_logisticreg(restaurant_features, restaurant_stars, restaurant_features_test, restaurant_stars_test):
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

    return (isTrth(results)), accuracy / (len(restaurant_stars_test))


if __name__ == '__main__':
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

    dim_test = 5
    for i in range(dim_test):
        X, Y = create_ds1(1000)
        Xtest, Ytest = create_ds1(100)
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

        X, Y = create_ds2(1000)
        Xtest, Ytest = create_ds2(100)
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

        X, Y = create_ds3(1000)
        Xtest, Ytest = create_ds3(100)
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

        X, Y = create_ds4(1000)
        Xtest, Ytest = create_ds4(100)
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
