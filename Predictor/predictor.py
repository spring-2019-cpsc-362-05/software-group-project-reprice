# Author: Mohammadreza Hajy Heydary
# =================================================================================================================
# This script contains all the necessary modules for predicting the property value for a given user input
# =================================================================================================================
from sklearn.externals import joblib
# =================================================================================================================
class StatePredictor:
    def __init__(self):
        self.__model = joblib.load("models/state_model.pkl")
        self.__encoder = joblib.load("data_maps/state_map.pkl")

    def predict(self, state, past_obs):
        instance = list()
        instance.append(self.__encoder.fit_transform([state]))
        instance.extend(past_obs)

        return self.__model.predict([instance])[0]

# =================================================================================================================
class CountyPredictor:
    def __init__(self):
        self.__model = joblib.load("models/county_model.pkl")
        self.__encoder = joblib.load("data_maps/county_map.pkl")

    def predict(self, county, past_obs):
        instance = list()
        instance.append(self.__encoder.fit_transform([county]))
        instance.extend(past_obs)

        return self.__model.predict([instance])[0]

# =================================================================================================================
class CityPredictor:
    def __init__(self):
        self.__model = joblib.load("models/city_model.pkl")
        self.__encoder = joblib.load("data_maps/city_map.pkl")

    def predict(self, city, past_obs):
        instance = list()
        instance.append(self.__encoder.fit_transform([city]))
        instance.extend(past_obs)

        return self.__model.predict([instance])[0]

# =================================================================================================================
class NeighborhoodPredictor:
    def __init__(self):
        self.__model = joblib.load("models/neighborhood_model.pkl")
        self.__encoder = joblib.load("data_maps/neighborhood_map.pkl")

    def predict(self, neighborhood, past_obs):
        instance = list()
        instance.append(self.__encoder.fit_transform([neighborhood]))
        instance.extend(past_obs)

        return self.__model.predict([instance])[0]

# =================================================================================================================
class Predictor:
    def __init__(self):
        self.__state_model = StatePredictor()
        self.__county_model = CountyPredictor()
        self.__city_model = CityPredictor()
        self.__neighborhood_model = NeighborhoodPredictor()
        self.__data_manager = None

    def predict(self, state, county, city, neighborhood, square_footage):
        pass


# =================================================================================================================
def main():
    # state_predict = StatePredictor()
    # print(state_predict.predict("California", [355, 356, 355]))

    # county_predict = CountyPredictor()
    # print(county_predict.predict("Los Angeles County", [427, 428, 429]))

    # city_predict = CityPredictor()
    # print(city_predict.predict("Fullerton", [395, 396, 396]))

    neighborhood_predict = NeighborhoodPredictor()
    # format [state, county, city, time series] -> 885
    print(neighborhood_predict.predict("Hollywood Hills", [355, 430, 474, 881, 882, 883]))


if __name__ == '__main__': main()
