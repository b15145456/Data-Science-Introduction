#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('C:/Users/user/Downloads/ENB2012_data.csv')
   
    feature_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    label_col = ['Y1']
    X = data[feature_col]
    y = data[label_col]

    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scaled = minmax_scaler.transform(X_train)
    X_test_scaled = minmax_scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def gradient_descent(X, y, alpha = .001, iters = 100000, eps=1e-4):
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    new_iters = 0
    new_theta = 0
    tmp = 0
    while new_iters < iters:
      for i in range(n):
        g = numpy.dot(X.T,( numpy.dot(X,theta)-y))+abs(eps*theta)
        new_theta = theta-alpha*g
        theta = new_theta
        new_iters += 1
    return theta


def predict(X, theta):
    return numpy.dot(X, theta)


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, 0, 1)

    theta = gradient_descent(X_train_scaled, y_train)
    y_hat = predict(X_train_scaled, theta)
    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scaled, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))


if __name__ == "__main__":
    main(sys.argv)


