# Author: Mohammadreza Hajy Heydary
# =================================================================================================================
# This script reads in the raw zip code-level data, preprocess it, and trains and random forest regression model on it
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
    df = pd.read_csv('data/Zip_MedianValuePerSqft_AllHomes.csv', encoding='latin-1')
    # drop extra columns
    df.drop(columns=['RegionID', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'], inplace=True)
    # Encode the region name
    lb_make = LabelEncoder()
    df["RegionName"] = lb_make.fit_transform(df["RegionName"])
    joblib.dump(lb_make, "data_maps/zip_map.pkl")

    # impute the missing values by using NEXT valid observation to fill gap
    df.fillna(method='bfill', axis=0, inplace=True)
    # impute the remaining missing values by propagating last valid observation forward to next valid backfill
    df.fillna(method='ffill', axis=0, inplace=True)
    # pass a temporal window of size m on the data and generate the target data set
    X = []
    Y = []
    for row_idx in range(0, len(df)):
        row = df.values[row_idx, :]
        # add the zip code to all instances
        for i in range(1, len(row) - (window_size + 1)):
            tmp = [row[0]]
            tmp.extend(row[i:(i + window_size + 1)])
            X.append(tmp)
            Y.append([row[i + window_size + 1]])
    # save the preprocessed data
    pd.DataFrame(X).to_csv("preprocessed_data/zip_X.csv", index=False, header=False)
    pd.DataFrame(Y).to_csv("preprocessed_data/zip_Y.csv", index=False, header=False)

# =================================================================================================================
def model_test():
    X = pd.read_csv("preprocessed_data/zip_X.csv", header=None).values
    Y = pd.read_csv("preprocessed_data/zip_Y.csv", header=None).values
    # print(X.shape)
    # print(Y.shape)
    kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)

    results = []
    for Num_trees in [50, 100, 200]:
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

    file_obj = open("tuning_results/zip_level.csv", 'w')
    file_obj.write("NumTrees,MaxDepth,train_RMSE,test_RMSE\n")
    for row in results:
        file_obj.write(row + '\n')

    file_obj.close()

# =================================================================================================================
def model_generate():
    X = pd.read_csv("preprocessed_data/zip_X.csv", header=None).values
    Y = pd.read_csv("preprocessed_data/zip_Y.csv", header=None).values
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=SEED, n_jobs=-1)
    model.fit(X, Y)
    joblib.dump(model, "models/zip_model.pkl")
    # joblib.load(filename)

# =================================================================================================================
def main():
    # preprocessor()
    # model_test()
    model_generate()


if __name__ == '__main__':main()
