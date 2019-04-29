# Author: Mohammadreza Hajy Heydary
# =================================================================================================================
# This script tests the performance of test module
# =================================================================================================================
from sklearn import metrics
from sklearn.externals import joblib
import Source.predictor as predict
import pandas as pd
import numpy as np
# =================================================================================================================
SEED = 23
# =================================================================================================================
def predictor_test():
    print("========================{}========================".format('Start of prediction test'))
    passed_ctr = 0
    total_tests = 0

    # *********************************************************
    print("Testing scalar methods functionality")
    predictor = predict.Predictor()
    total_tests += 1
    if predictor.scale(0 * 100) == 0:
        print("--> test 1 passed")
        passed_ctr += 1
    else:
        print("--> test 1 failed")

    total_tests += 1
    if predictor.scale(120, 1000) == 120000:
        print("--> test 2 passed")
        passed_ctr += 1
    else:
        print("--> test 2 failed")

    # *********************************************************
    print("Testing state level predictors performance on the test dataset")
    test_df_X = pd.read_csv('preprocessed_data/state_X.csv')
    test_df_Y = pd.read_csv('preprocessed_data/state_Y.csv')
    sampled_X = test_df_X.sample(n=100, replace=False, random_state=SEED).values
    sampled_Y = test_df_Y.sample(n=100, replace=False, random_state=SEED).values.flatten()
    state_predictor = predict.StatePredictor()
    encoder = joblib.load("data_maps/state_map.pkl")
    state = encoder.inverse_transform(sampled_X[:, 0].astype('int'))
    results = []
    for i in range(0, len(state)):
        results.append(state_predictor.predict(state[i], sampled_X[i, 1:]))

    total_tests += 1
    if metrics.mean_squared_error(y_true=sampled_Y, y_pred=results) <= 3:
        print("--> test 1 passed")
        passed_ctr += 1
    else:
        print("--> test 2 failed")

    # *********************************************************
    print("Testing county level predictors performance on the test dataset")
    test_df_X = pd.read_csv('preprocessed_data/county_X.csv')
    test_df_Y = pd.read_csv('preprocessed_data/county_Y.csv')
    sampled_X = test_df_X.sample(n=100, replace=False, random_state=SEED).values
    sampled_Y = test_df_Y.sample(n=100, replace=False, random_state=SEED).values.flatten()
    county_predictor = predict.CountyPredictor()
    encoder = joblib.load("data_maps/county_map.pkl")
    county = encoder.inverse_transform(sampled_X[:, 0].astype('int'))
    results = []
    for i in range(0, len(county)):
        results.append(county_predictor.predict(county[i], sampled_X[i, 1:]))

    total_tests += 1
    if metrics.mean_squared_error(y_true=sampled_Y, y_pred=results) <= 3:
        print("--> test 1 passed")
        passed_ctr += 1
    else:
        print("--> test 2 failed")

    print("========================{}========================".format('End of prediction test'))
    print("{} out of {} passed the test\n\n".format(passed_ctr, total_tests))

    # *********************************************************
    print("Testing city level predictors performance on the test dataset")
    test_df_X = pd.read_csv('preprocessed_data/city_X.csv')
    test_df_Y = pd.read_csv('preprocessed_data/city_Y.csv')
    sampled_X = test_df_X.sample(n=100, replace=False, random_state=SEED).values
    sampled_Y = test_df_Y.sample(n=100, replace=False, random_state=SEED).values.flatten()
    city_predictor = predict.CityPredictor()
    encoder = joblib.load("data_maps/city_map.pkl")
    city = encoder.inverse_transform(sampled_X[:, 0].astype('int'))
    results = []
    for i in range(0, len(city)):
        results.append(city_predictor.predict(city[i], sampled_X[i, 1:]))

    total_tests += 1
    if metrics.mean_squared_error(y_true=sampled_Y, y_pred=results) <= 3:
        print("--> test 1 passed")
        passed_ctr += 1
    else:
        print("--> test 2 failed")

    print("========================{}========================".format('End of prediction test'))
    print("{} out of {} passed the test\n\n".format(passed_ctr, total_tests))

    # *********************************************************


def main():
    predictor_test()


if __name__ == '__main__': main()
