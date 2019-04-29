# Author: Mohammadreza Hajy Heydary
# =================================================================================================================
# This script contains all the necessary modules for predicting the property value for a given user input
# =================================================================================================================
from sklearn.externals import joblib
from Source.input import DataManagement
# =================================================================================================================
class StatePredictor:
    def __init__(self):
        self.__model = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/models/state_model.pkl")
        self.__encoder = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/data_maps/state_map.pkl")

    def predict(self, state, past_obs):
        instance = list()
        instance.append(self.__encoder.fit_transform([state]))
        instance.extend(past_obs)

        return self.__model.predict([instance])[0]

# =================================================================================================================
class CountyPredictor:
    def __init__(self):
        self.__model = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/models/county_model.pkl")
        self.__encoder = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/data_maps/county_map.pkl")

    def predict(self, county, past_obs):
        instance = list()
        instance.append(self.__encoder.fit_transform([county]))
        instance.extend(past_obs)

        return self.__model.predict([instance])[0]

# =================================================================================================================
class CityPredictor:
    def __init__(self):
        self.__model = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/models/city_model.pkl")
        self.__encoder = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/data_maps/city_map.pkl")

    def predict(self, city, past_obs):
        instance = list()
        instance.append(self.__encoder.fit_transform([city]))
        instance.extend(past_obs)

        return self.__model.predict([instance])[0]

# =================================================================================================================
class NeighborhoodPredictor:
    def __init__(self):
        self.__model = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/models/neighborhood_model.pkl")
        self.__encoder = joblib.load("/Users/reza/Documents/Spring 2019/SE 362/Project/Source/data_maps/neighborhood_map.pkl")

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
        self.__data_manager = DataManagement()

    @staticmethod
    def scale(price, square_footage):
        return price * square_footage

    def predict(self, state, county, city, neighborhood, square_footage):
        # retrieve the most recent values for the current entries
        state_prediction = self.__state_model.predict(state, self.__data_manager.getState(state))
        county_prediction = self.__county_model.predict(county, self.__data_manager.getCounty(county))
        city_prediction = self.__city_model.predict(city, self.__data_manager.getCity(city))
        nb_extended = [state_prediction, county_prediction, city_prediction]
        nb_extended.extend(self.__data_manager.getNeighborhood(neighborhood))
        neighborhood_prediction = self.__neighborhood_model.predict(neighborhood, nb_extended)

        return self.scale(neighborhood_prediction, square_footage)


# =================================================================================================================
def main():
    # state_predict = StatePredictor()
    # print(state_predict.predict("California", [355, 356, 355]))

    # county_predict = CountyPredictor()
    # print(county_predict.predict("Los Angeles County", [427, 428, 429]))

    # city_predict = CityPredictor()
    # print(city_predict.predict("Fullerton", [395, 396, 396]))

    # neighborhood_predict = NeighborhoodPredictor()
    # format [state, county, city, time series] -> 885
    # print(neighborhood_predict.predict("Hollywood Hills", [355, 430, 474, 881, 882, 883]))

    value_predict = Predictor()
    price = value_predict.predict(state="California", county="Orange County", city="Anaheim", neighborhood="Southwest Anaheim", square_footage=1000)
    print(price)


if __name__ == '__main__': main()
