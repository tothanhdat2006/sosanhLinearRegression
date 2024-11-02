import data

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import keras

def LinearRegression_model():
    return make_pipeline(data.preprocessing_pipeline(), LinearRegression())


def RandomForest_model():
    return make_pipeline(data.preprocessing_pipeline(), RandomForestRegressor(random_state=86))


def DNN_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2,)))
    model.add(keras.layers.Dense(200, kernel_initializer="normal", activation="relu"))
    model.add(keras.layers.Dense(200, kernel_initializer="normal", activation="relu"))
    model.add(keras.layers.Dense(200, kernel_initializer="normal", activation="relu"))
    model.add(keras.layers.Dense(200, kernel_initializer="normal", activation="relu"))
    model.add(keras.layers.Dense(1, kernel_initializer="normal", activation="linear"))
    return model
