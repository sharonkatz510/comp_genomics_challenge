# This module takes care of feature selection
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def spear_corr(x, y):
    X = pd.Series(x)
    Y = pd.Series(y)
    return X.corr(Y, method="spearman")


def ffs(k: "number of features", data: "pandas DataFrame", label, fn: "Feature evaluation function" = spear_corr):
    """This function implements ffs using multi-feature regression and evaluation
    function fn"""
    data["free_var"] = np.ones(len(data))
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    best_features = 'free_var',
    for i in range(k):
        available_features = list(data.columns.drop(labels=list(best_features)))
        feature_scores = {}
        for feature_name in available_features:
            feature_list = list(best_features)
            feature_list.append(feature_name)
            feature_comb = x_train[feature_list]
            reg = LinearRegression().fit(feature_comb, y_train)
            feature_scores[feature_name] = spear_corr(reg.predict(x_test[feature_list]), y_test)
        best_features = best_features + (max(feature_scores, key=feature_scores.get),)
    return best_features
