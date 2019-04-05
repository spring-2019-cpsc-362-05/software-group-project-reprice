# Author: Mohammadreza Hajy Heydary
# =================================================================================================================
# This script reads in the raw city-level data, preprocess it, and trains and random forest regression model on it
# =================================================================================================================
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
# =================================================================================================================
window_size = 3
SEED = 23
num_folds = 4
# =================================================================================================================
def preprocessor():
    df = pd.read_csv('data/City_MedianValuePerSqft_AllHomes.csv', encoding='latin-1')
    # drop extra columns
    df.drop(columns=['RegionID', 'State', 'Metro', 'CountyName', 'SizeRank'], inplace=True)

    # Encode the region name
    lb_make = LabelEncoder()
    df["RegionName"] = lb_make.fit_transform(df["RegionName"])
    joblib.dump(lb_make, "data_maps/city_map.pkl")

    # impute the missing values by using NEXT valid observation to fill gap
    df.fillna(method='bfill', axis=0, inplace=True)

    # pass a temporal window of size m on the data and generate the target data set
    X = []
    Y = []
    for row_idx in range(0, len(df)):
        row = df.values[row_idx, :]
        # add the city code to all instances
        for i in range(1, len(row) - window_size):
            tmp = [row[0]]
            tmp.extend(row[i:(i + window_size)])
            X.append(tmp)
            Y.append([row[i + window_size]])
    # save the preprocessed data
    pd.DataFrame(X).to_csv("preprocessed_data/city_X.csv", index=False, header=False)
    pd.DataFrame(Y).to_csv("preprocessed_data/city_Y.csv", index=False, header=False)

# =================================================================================================================
def model_test():
    X = pd.read_csv("preprocessed_data/city_X.csv", header=None).values
    Y = pd.read_csv("preprocessed_data/city_Y.csv", header=None).values
    kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)

    results = []
    for Num_trees in [50, 100, 200, 300]:
        for depth in [10, 20, 40, None]:
            train_error = []
            test_error = []
            for train_index, test_index in kf.split(X):
                model = RandomForestRegressor(n_estimators=Num_trees, max_depth=depth, random_state=SEED, n_jobs=-1)
                model.fit(X[train_index], Y[train_index])

                train_error.append(metrics.mean_squared_error(model.predict(X[train_index]), Y[train_index]) ** 0.5)
                test_error.append(metrics.mean_squared_error(model.predict(X[test_index]), Y[test_index]) ** 0.5)
                print("Train Error: {}".format(train_error[-1]))
                print("Test Error: {}".format(test_error[-1]))

            results.append("{},{},{},{}".format(Num_trees, depth, np.mean(train_error), np.mean(test_error)))

    file_obj = open("tuning_results/city_level.csv", 'w')
    file_obj.write("NumTrees,MaxDepth,train_RMSE,test_RMSE\n")
    for row in results:
        file_obj.write(row + '\n')

    file_obj.close()

# =================================================================================================================
def model_generate():
    X = pd.read_csv("preprocessed_data/city_X.csv", header=None).values
    Y = pd.read_csv("preprocessed_data/city_Y.csv", header=None).values
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=SEED, n_jobs=-1)
    model.fit(X, Y)
    joblib.dump(model, "models/city_model.pkl", compress=True)

# =================================================================================================================
def main():
    # preprocessor()
    # model_test()
    model_generate()


if __name__ == '__main__':main()
