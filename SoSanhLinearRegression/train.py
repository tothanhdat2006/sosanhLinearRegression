import data
import models
import config

import tensorflow as tf
import keras
import dill
from keras import ops


def save_model(final_model, final_model_name):
    with open(config.PATH_model + "/" + final_model_name + ".pkl", "wb") as file:
        dill.dump(final_model, file)


def training_LinearRegression():
    X_train, X_test, y_train, y_test = data.split_data()
    lin_reg = models.LinearRegression_model()
    print("Training Linear Regression...")
    lin_reg.fit(X_train, y_train)
    save_model(lin_reg, "LinearRegression_model")


def training_RandomForest():
    X_train, X_test, y_train, y_test = data.split_data()
    randForest = models.RandomForest_model()
    print("Training Random Forest...")
    randForest.fit(X_train, y_train)
    save_model(randForest, "RandomForest_model")


@keras.saving.register_keras_serializable(package="RMSELoss", name="root_mean_squared_error")
def root_mean_squared_error(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true)))


def training_DNN():
    X_train, X_test, y_train, y_test = data.split_preprocess_data()
    X_train = tf.convert_to_tensor(X_train.toarry() if hasattr(X_train, 'toarray') else X_train)
    DNN_model = models.DNN_model()
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    metric = keras.metrics.RootMeanSquaredError()
    DNN_model.compile(loss=root_mean_squared_error, optimizer=optimizer, metrics=[metric])
    print("Training DNN...")
    DNN_model.fit(X_train, y_train, epochs=50, validation_split=0.2)
    save_model(DNN_model, "DNN_model")
