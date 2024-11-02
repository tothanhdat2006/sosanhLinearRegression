import sys
import config

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split


def preprocessing_pipeline():
    num_pipeline = Pipeline([
        ("standardize", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("onehotencode", OneHotEncoder()),
    ])
    return make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )


def read_data():
    return pd.read_csv(config.PATH_dataset + "/Housing.csv")


def split_data():
    housing_data = read_data()
    X = housing_data.drop(columns=["price"], axis=1)
    X = X[["area", "bedrooms"]]
    y = housing_data["price"] / 1000.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 86)
    return X_train, X_test, y_train, y_test


def split_preprocess_data():
    housing_data = read_data()
    X = housing_data.drop(columns=["price"], axis=1)
    X = X[["area", "bedrooms"]]
    X_preprocess = preprocessing_pipeline().fit_transform(X)
    y = housing_data["price"] / 1000.0
    X_train, X_test, y_train, y_test = train_test_split(X_preprocess, y, test_size=0.2, random_state=86)
    return X_train, X_test, y_train, y_test

# print(X_train.head(5))
# print(y_train.head(5))
# print(X_test.shape)
# print(y_test.shape)