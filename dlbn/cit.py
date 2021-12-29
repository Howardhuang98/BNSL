#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   cit.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/25 18:29  
------------      
"""
from math import sqrt, log

import numpy as np
import pandas as pd
from scipy.stats import norm


def fisherz(data, X, Y, condition_set, correlation_matrix=None):
    '''
    Perform an independence test using Fisher-Z's test
    Parameters
    ----------
    data : data matrices
    X: int
    Y: int
    condition_set: set
    correlation_matrix : correlation matrix;
                         None means without the parameter of correlation matrix
    Returns
    -------
    p : the p-value of the test
    '''
    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data.T)
    sample_size = data.shape[0]
    condition_set = tuple(condition_set)
    var = list((X, Y) + condition_set)
    sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(sample_size - len(condition_set) - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(X)))
    return p


if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    fisherz(data.values,0,1,(2,3))
