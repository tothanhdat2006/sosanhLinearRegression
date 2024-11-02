import config
import data

import dill
import tensorflow as tf

def load_saved_model(final_model_name):
    with open(config.PATH_model + "/" + final_model_name + ".pkl", "rb") as file:
        return dill.load(file)


def predict_LinearRegression():
    X_train, X_test, y_train, y_test = data.split_data()
    model = load_saved_model("LinearRegression_model")
    # print("-----------------Linear Regression Predictions-------------------")
    y_pred = model.predict(X_test)
    # print("-----------------Real value-------------------")
    # print(y_test)


def predict_RandomForest():
    X_train, X_test, y_train, y_test = data.split_data()
    model = load_saved_model("RandomForest_model")
    # print("-----------------Random Forest Predictions-------------------")
    y_pred = model.predict(X_test)
    # print("-----------------Real value-------------------")
    # print(y_test)


def predict_DNN():
    X_train, X_test, y_train, y_test = data.split_preprocess_data()
    X_test = tf.convert_to_tensor(X_test.toarry() if hasattr(X_test, 'toarray') else X_test)
    model = load_saved_model("DNN_model")
    # print("-----------------DNN Predictions-------------------")
    y_pred = model.predict(X_test)
    # print("-----------------Real value-------------------")
    # print(y_test)
