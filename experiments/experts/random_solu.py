#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   random_solu.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/4/12 21:10  
------------      
"""
import numpy as np

from bnsl import graph
from bnsl.graph import DAG

if __name__ == '__main__':
    true_dag = DAG()
    true_dag.read(r"../data/Alarm/alarm_net.csv")
    result = []
    for i in range(1000):
        a = np.random.rand(37,37)
        fnd = np.linalg.norm(true_dag.adj_np - a)
        result.append(fnd)
    print(np.average(result))
