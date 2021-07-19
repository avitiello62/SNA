import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import sys
#LOCAL IMPORT
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


def logistic_regression_train(restaurant_features, restaurant_stars):
    """
    :param restaurant_features:
    :param restaurant_stars:
    :return:
    """
    log_reg = LogisticRegression()
    log_reg.fit(restaurant_features, restaurant_stars)
    return log_reg



def linear_regressor_train(restaurant_features, restaurant_stars):
    """
    :param restaurant_features:
    :param restaurant_stars:
    :return:
    """
    lin_reg = LinearRegression(positive=True)
    lin_reg.fit(restaurant_features, restaurant_stars)
    return lin_reg
    


def logistic_regressor_ic_train(restaurant_features, restaurant_stars):
    """
    :param restaurant_features:
    :param restaurant_stars:
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
    return log_reg
    

def logistic_classifier_test(classifier,restaurant_features_test, restaurant_stars_test):
    results = {}
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                key = (food, service, value)
                results[key] = classifier.predict([key])
    accuracy = 0
    for i in range(len(restaurant_features_test)):
        if classifier.predict([restaurant_features_test[i]]) == restaurant_stars_test[i]:
            accuracy += 1
    accuracy=accuracy/len(restaurant_stars_test)
    is_truthful=isTruthful(results)
    return is_truthful, accuracy

def linear_classifier_test(classifier,restaurant_features_test, restaurant_stars_test):
    results = {}
    for food in range(0, 6):
        for service in range(-1, 6):
            for value in range(-1, 6):
                key = (food, service, value)
                results[key] = np.round(classifier.predict([key]))
    accuracy = 0
    for i in range(len(restaurant_features_test)):
        if np.round(classifier.predict([restaurant_features_test[i]])) == restaurant_stars_test[i]:
            accuracy += 1
    accuracy=accuracy/len(restaurant_stars_test)
    is_truthful=isTruthful(results)
    
    return is_truthful, accuracy

def complete_set_creation(dim):
    dataset={}
    dataset['coefficient']=coefficient_based_dataset(dim)
    dataset['max']=max_based_dataset(dim)
    dataset['average']=average_based_dataset(dim)
    dataset['random']=totally_random_dataset(dim)
    return dataset

def performance_evaluation():
    

    
    dim_training_set=10000
    dim_test_set=1000
    training_sets=complete_set_creation(dim_training_set)
    test_sets=complete_set_creation(dim_test_set)
    label=['coefficient','max','average','random']
    result={}
    

    for tr_key in label:
        result[tr_key]={}
        classifier='Logistic Regressor'
        log_reg_trained=logistic_regression_train(training_sets[tr_key][0],training_sets[tr_key][1])
        print("{} trained on {} with {} samples".format(classifier,tr_key,dim_training_set))
        for test_key in label:
            result[tr_key][test_key]=logistic_classifier_test(log_reg_trained,test_sets[test_key][0],test_sets[test_key][1])
            print("\t{} Dataset with {} samples".format(test_key,dim_test_set))
            print("\tAccuracy: {} Is Truthful?: {}".format( result[tr_key][test_key][1], result[tr_key][test_key][0]))
    

    
    for tr_key in label:
        result[tr_key]={}
        classifier='Logistic Regressor IC'
        log_reg_ic_trained=logistic_regressor_ic_train(training_sets[tr_key][0],training_sets[tr_key][1])
        print("{} trained on {} with {} samples".format(classifier,tr_key,dim_training_set))
        for test_key in label:
            result[tr_key][test_key]=logistic_classifier_test(log_reg_ic_trained,test_sets[test_key][0],test_sets[test_key][1])
            print("\t{} Dataset with {} samples".format(test_key,dim_test_set))
            print("\tAccuracy: {} Is Truthful?: {}".format( result[tr_key][test_key][1], result[tr_key][test_key][0]))
    for tr_key in label:
        result[tr_key]={}
        classifier='Linear Regressor'
        lin_reg_trained=linear_regressor_train(training_sets[tr_key][0],training_sets[tr_key][1])
        print("{} trained on {} with {} samples".format(classifier,tr_key,dim_training_set))
        for test_key in label:
            result[tr_key][test_key]=linear_classifier_test(lin_reg_trained,test_sets[test_key][0],test_sets[test_key][1])
            print("\t{} Dataset with {} samples".format(test_key,dim_test_set))
            print("\tAccuracy: {} Is Truthful?: {}".format( result[tr_key][test_key][1], result[tr_key][test_key][0]))
        
if __name__ == '__main__':
    performance_evaluation()
