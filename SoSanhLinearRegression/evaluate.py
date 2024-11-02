import config
import data

import dill
import tensorflow as tf
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

def load_saved_model(final_model_name):
    with open(config.PATH_model + "/" + final_model_name + ".pkl", "rb") as file:
        return dill.load(file)


def eval_LinearRegression():
    X_train, X_test, y_train, y_test = data.split_data()
    lin_reg = load_saved_model("LinearRegression_model")
    pred = lin_reg.predict(X_test)
    print("Linear Regression RMSE score: ", root_mean_squared_error(y_test, pred))
    # print("Linear Regression R^2 score: ", r2_score(y_test, pred))

def eval_RandomForest():
    X_train, X_test, y_train, y_test = data.split_data()
    randForest = load_saved_model("RandomForest_model")
    pred = randForest.predict(X_test)
    print("Random Forest RMSE score: ", root_mean_squared_error(y_test, pred))

def eval_DNN():
    X_train, X_test, y_train, y_test = data.split_preprocess_data()
    X_test = tf.convert_to_tensor(X_test.toarry() if hasattr(X_test, 'toarray') else X_test)
    DNN = load_saved_model("DNN_model")
    print("DNN RMSE score ([loss, accuracy]): ", DNN.evaluate(X_test, y_test))
