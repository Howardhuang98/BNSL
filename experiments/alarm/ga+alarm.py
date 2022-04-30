#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ga+hailfinder.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/1/3 17:59  
------------      
"""
import pandas as pd

from bnsl.estimators import GA

for n in [50, 500, 2000, 5000]:
    print("****{}****".format(n))
    data = pd.read_csv(r"../data/Alarm/alarm.csv")[:n]
    ga = GA(data)
    dag = ga.run(max_iter=1000, pop=50)
    dag.to_csv_adj(r"./result/ga+alarm{}.csv".format(n))
