#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   learning.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/5/5 20:20  
------------      
"""
import copy
import sys

import pandas as pd

sys.path.append("../")
from bnsl.graph import compare
import numpy as np

import bnsl

if __name__ == '__main__':
    result = pd.DataFrame()
    alarm_dataset = bnsl.Dataset("alarm")
    g = alarm_dataset.dag
    adj = g.adj_df(g.nodes)
    did = {}
    for i in range(500):
        x, y = np.random.randint(0, len(g.nodes), size=2)
        while (x, y) in did:
            x, y = np.random.randint(0, len(g.nodes), size=2)
        adj.iloc[x][y] = adj.iloc[x][y] ^ 1
        print(np.linalg.norm(g.adj_df(g.nodes).values-adj.values,ord='f'))
        est = bnsl.L2C(alarm_dataset.data[:2000], [adj], [1])
        est.run(restart=5)
        print(compare(est.result, alarm_dataset.dag))
        row = result.shape[0]
        result.loc[row, "Norm of E"] = np.linalg.norm(g.adj_df(g.nodes).values-adj.values,ord='f')
        result.loc[row, "Norm of result"] = compare(est.result, alarm_dataset.dag)['norm']
        result.to_csv(r"./result/learning_ability.csv")
