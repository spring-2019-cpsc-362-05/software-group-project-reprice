# Author: Mohammadreza Hajy Heydary
# =================================================================================================================
# This script reads in the raw neighborhood-level data, preprocess it, and trains and random forest regression model on it
# =================================================================================================================
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
# =================================================================================================================
STATE_ABBREV = {
    'AL':'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR':'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
}
# =================================================================================================================
window_size = 3
SEED = 23
num_folds = 4
# =================================================================================================================
def preprocessor():
    # read all the  necessary data
    state_df = pd.read_csv('data/State_MedianValuePerSqft_AllHomes.csv')
    city_df = pd.read_csv('data/City_MedianValuePerSqft_AllHomes.csv', encoding='latin-1')
    county_df = pd.read_csv('data/County_MedianValuePerSqft_AllHomes.csv', encoding='latin-1')
    nb_df = pd.read_csv('data/Neighborhood_MedianValuePerSqft_AllHomes.csv')

    # drop extra columns in neighborhood df
    nb_df.drop(columns=['RegionID'], inplace=True)

    # Encode the region name
    lb_make = LabelEncoder()
    nb_df["RegionName"] = lb_make.fit_transform(nb_df["RegionName"])
    joblib.dump(lb_make, "data_maps/neighborhood_map.pkl")

    # impute all the missing values backward fill
    state_df.fillna(method='bfill', axis=0, inplace=True)
    city_df.fillna(method='bfill', axis=0, inplace=True)
    county_df.fillna(method='bfill', axis=0, inplace=True)
    nb_df.fillna(method='bfill', axis=0, inplace=True)
    # forward fill
    state_df.fillna(method='ffill', axis=0, inplace=True)
    city_df.fillna(method='ffill', axis=0, inplace=True)
    county_df.fillna(method='ffill', axis=0, inplace=True)
    nb_df.fillna(method='ffill', axis=0, inplace=True)

    # pass a temporal window of size m on the data and generate the target data set
    X = []
    Y = []
    ctr = 0
    for index, row in nb_df.iterrows():
        value_row = row.iloc[6:]
        for i in range(0, len(value_row) - window_size):
            try:
                # initial values that shall be added to each instance
                # NB code
                time = value_row.axes[0][i + window_size]
                tmp = [row.iloc[0]]
                # state median property value at time [i + window_size]
                tmp.extend(state_df[state_df["RegionName"] == STATE_ABBREV[row.iloc[2]]][time])
                # county median property value at time [i + window_size]
                t = county_df[(county_df["RegionName"] == row.iloc[4]) & (county_df["State"] == row.iloc[2])][time]
                if len(t == 1):
                    tmp.extend(t)
                else:
                    ctr += 1
                    continue

                # city median property value at time [i + window_size]
                t = city_df[(city_df["RegionName"] == row.iloc[1]) & (city_df["State"] == row.iloc[2]) & (city_df["CountyName"] == row.iloc[4])][time]
                # print(t)
                if len(t == 1):
                    tmp.extend(t)
                else:
                    ctr += 1
                    continue

                tmp.extend(value_row[i:(i + window_size)])
                X.append(tmp)
                Y.append([value_row[i + window_size]])
            except KeyError:
                print("A key was not found. The instance shall be ignored.")
                continue

    print("Number of instances ignore due to value errors: {}".format(ctr))
    # make sure there is nan
    X = pd.DataFrame(X).dropna(axis=1)
    Y = pd.DataFrame(Y)
    # save the preprocessed data
    X.to_csv("preprocessed_data/neighborhood_X.csv", index=False, header=False)
    Y.to_csv("preprocessed_data/neighborhood_Y.csv", index=False, header=False)

# =================================================================================================================
def model_test():
    X = pd.read_csv("preprocessed_data/neighborhood_X.csv", header=None).dropna(axis=1).values
    Y = pd.read_csv("preprocessed_data/neighborhood_Y.csv", header=None).values
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

    file_obj = open("tuning_results/neighborhood_level.csv", 'w')
    file_obj.write("NumTrees,MaxDepth,train_RMSE,test_RMSE\n")
    for row in results:
        file_obj.write(row + '\n')

    file_obj.close()

# =================================================================================================================
def model_generate():
    X = pd.read_csv("preprocessed_data/neighborhood_X.csv", header=None).dropna(axis=1).values
    Y = pd.read_csv("preprocessed_data/neighborhood_Y.csv", header=None).values
    model = RandomForestRegressor(n_estimators=200, max_depth=40, random_state=SEED, n_jobs=-1)
    model.fit(X, Y)
    joblib.dump(model, "models/neighborhood_model.pkl", compress=True)

# =================================================================================================================
def main():
    # preprocessor()
    # model_test()
    model_generate()


if __name__ == '__main__':main()