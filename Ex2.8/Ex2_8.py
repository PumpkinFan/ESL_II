#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:04:54 2023

@author: charlesc
"""

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

TRAIN_FILENAME = "all_digits.train"
TEST_FILENAME = "all_digits.test"

def load_twos_threes_data(filename):
    with open(filename) as fn:
        data = np.loadtxt(fn, dtype=float)
    twos_data = [d[1:] for d in data if d[0] == 2]
    threes_data = [d[1:] for d in data if d[0] == 3]
    return twos_data, threes_data


def create_targets(*data_vectors):
    targets = np.array([])
    for i, dv in enumerate(data_vectors):
        trg = i + np.zeros(len(dv))
        targets = np.concatenate((targets, trg, ))
    return targets


def score_binary_model(model, data0, data1):
    r0 = model.predict(data0)
    r1 = model.predict(data1)
    
    r0_wrong = np.count_nonzero(r0 >= 0.5)
    r1_wrong = np.count_nonzero(r1 < 0.5)
    
    return (r0_wrong + r1_wrong) / (r0.shape[0] + r1.shape[0])


if __name__ == "__main__":
    twos_train, threes_train = load_twos_threes_data(TRAIN_FILENAME)
    twos_test, threes_test = load_twos_threes_data(TEST_FILENAME)
    
    # k-nearest neighbors
    ks_to_try = [1, 3, 5, 7, 15]
    train_errors = []
    test_errors = []
    
    train_data = np.concatenate((twos_train, threes_train, ))
    train_targets = create_targets(twos_train, threes_train)
    test_data = np.concatenate((twos_test, threes_test, ))
    test_targets = create_targets(twos_test, threes_test)    
    
    for k in ks_to_try:
        kn_clf = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        kn_clf.fit(train_data, train_targets)
        
        # train_error = 1 - kn_clf.score(train_data, train_targets)
        train_error = score_binary_model(kn_clf, twos_train, threes_train)
        train_errors.append(train_error)
        # test_error = 1 - kn_clf.score(test_data, test_targets)
        test_error = score_binary_model(kn_clf, twos_test, threes_test)
        test_errors.append(test_error)
    
    # linear regression
    lin_reg = LinearRegression().fit(train_data, train_targets)
    # reg_train_error = 1 - lin_reg.score(train_data, train_targets)
    # reg_test_error = 1 - lin_reg.score(test_data, test_targets)
    reg_train_error = score_binary_model(lin_reg, twos_train, threes_train)
    reg_test_error = score_binary_model(lin_reg, twos_test, threes_test)
    
    
    plt.plot(ks_to_try, train_errors, 'o', c='darkblue')
    plt.plot(ks_to_try, test_errors, 'o', c='firebrick')
    plt.axhline(y=reg_train_error, c="lightblue")
    plt.axhline(y=reg_test_error, c="lightcoral")
        
        