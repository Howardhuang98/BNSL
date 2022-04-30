#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   asian.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/4/11 21:28  
------------      
"""
import copy
import warnings

import numpy as np
import pandas as pd

from bnsl import graph
from bnsl.estimators import KBNL
from bnsl.graph import DAG, acc

warnings.filterwarnings("ignore")


def experts_generater(n_mistakes):
    true_dag = DAG()
    true_dag.read(r"../data/Alarm/alarm_net.csv")
    e_df = true_dag.adj_df
    n = len(e_df.columns)
    res = []
    for j in range(4):
        x = 0
        y = 0
        new_e = copy.deepcopy(e_df)
        for k in range(n_mistakes):
            while x == y:
                x = np.random.choice([i for i in range(n)])
                y = np.random.choice([i for i in range(n)])
            new_e.iloc[x, y] = 1 ^ int(e_df.iloc[x, y])
            x = 0
            y = 0
        res.append(new_e)
    return res


def run():
    true_dag = DAG()
    true_dag.read(r"../data/Alarm/alarm_net.csv")
    data = pd.read_csv(r"../data/Alarm/alarm.csv")[:2000]
    result = []
    for n in range(1, 5):
        print(f"**** Expert with {n} errors ****")
        kbnl = KBNL(data, experts_generater(n), [0.25, 0.25, 0.25, 0.25])
        kbnl.run_parallel(worker=10, restart=1, explore_num=8, num_parents=5)
        kbnl.result.to_csv_adj(rf"./result/Experts with {n} errors.csv")
        norm, shd, dag_norm = kbnl.expert.norm_distance(true_dag), true_dag - kbnl.result, graph.norm_distance(
            kbnl.result, true_dag)
        print(true_dag.adj_df)
        print(kbnl.result.adj_df)
        print("Expert norm distance", norm)
        print("DAG norm distance", dag_norm)
        print("DAG shd", shd)
        result.append((norm, shd, dag_norm))
        with open("checkpoint.csv", 'a') as f:
            f.write(f"{norm}, {dag_norm}, {shd}\n")
    print(result)


if __name__ == '__main__':
    run()
